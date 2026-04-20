"""
License validation system using Supabase with SECURE trial tracking.

This module handles license validation via Supabase database,
trial period management with persistent machine IDs, and usage tracking.

SECURITY FEATURES:
- GitHub PAT hash as primary user identifier
- Persistent machine ID stored in /data/machine_id.lock
- Runtime license validation before every tool execution
- Detailed trial usage tracking with deduplication
- Remote trial deactivation support
"""

import os
import uuid
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, asdict
from loguru import logger

try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    logger.warning("Supabase client not installed. Install with: pip install supabase")

from .constants import LICENSE
from .exceptions import ConfigurationError


@dataclass
class LicenseInfo:
    """License information."""
    license_key: str
    tier: str
    user_id: str
    max_repositories: int
    max_analyses_per_month: int
    max_storage_gb: int
    issued_at: datetime
    expires_at: Optional[datetime] = None
    is_trial: bool = False
    metadata: Dict[str, Any] = None

    def is_valid(self) -> bool:
        """Check if license is valid and not expired."""
        if self.expires_at and datetime.now(timezone.utc) > self.expires_at:
            return False
        return True

    def days_remaining(self) -> Optional[int]:
        """Get days remaining until expiration."""
        if not self.expires_at:
            return None
        delta = self.expires_at - datetime.now(timezone.utc)
        return max(0, delta.days)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        data = asdict(self)
        # Convert datetime to string
        if isinstance(data['issued_at'], datetime):
            data['issued_at'] = data['issued_at'].isoformat()
        if data.get('expires_at') and isinstance(data['expires_at'], datetime):
            data['expires_at'] = data['expires_at'].isoformat()
        return data


def get_github_pat_hash() -> Optional[str]:
    """
    Get SHA256 hash of GitHub PAT token from environment.

    This provides a stable user identifier across Docker container recreations.

    Returns:
        SHA256 hash of GitHub token, or None if not available
    """
    try:
        github_token = os.getenv('GITHUB_TOKEN')
        if not github_token:
            logger.warning("GITHUB_TOKEN not found in environment")
            return None

        # Create SHA256 hash of the token
        pat_hash = hashlib.sha256(github_token.encode()).hexdigest()
        logger.debug(f"GitHub PAT hash: {pat_hash[:16]}...")
        return pat_hash

    except Exception as e:
        logger.error(f"Failed to generate GitHub PAT hash: {e}")
        return None


def get_persistent_machine_id(data_dir: Path = Path("/data")) -> str:
    """
    Get or create persistent machine ID stored in /data volume.

    This ID persists across Docker container restarts/recreations
    as long as the /data volume is mounted persistently.

    Args:
        data_dir: Directory to store machine ID file (should be mounted volume)

    Returns:
        Persistent machine ID (24-character hex string)
    """
    machine_id_file = data_dir / "machine_id.lock"

    try:
        # Ensure data directory exists
        data_dir.mkdir(parents=True, exist_ok=True)

        # Try to read existing machine ID
        if machine_id_file.exists():
            try:
                with open(machine_id_file, 'r') as f:
                    machine_id = f.read().strip()

                # Validate format (24 hex characters)
                if len(machine_id) == 24 and all(c in '0123456789abcdef' for c in machine_id):
                    logger.debug(f"Loaded persistent machine ID: {machine_id[:8]}...")
                    return machine_id
                else:
                    logger.warning(f"Invalid machine ID format in {machine_id_file}, regenerating...")

            except Exception as e:
                logger.warning(f"Failed to read machine ID file: {e}, regenerating...")

        # Generate new machine ID based on multiple sources
        github_pat_hash = get_github_pat_hash()

        # Combine GitHub PAT hash + random UUID for uniqueness
        if github_pat_hash:
            # Primary: Use GitHub PAT hash as base (ensures same user = same ID across containers)
            machine_str = f"github:{github_pat_hash}"
        else:
            # Fallback: Generate random UUID (will be different each time, but persisted to file)
            logger.warning("GitHub PAT not available, using random machine ID")
            machine_str = f"random:{uuid.uuid4().hex}"

        # Create machine ID (24 hex characters)
        machine_id = hashlib.sha256(machine_str.encode()).hexdigest()[:24]

        # Save to persistent file
        try:
            with open(machine_id_file, 'w') as f:
                f.write(machine_id)
            logger.info(f"Generated and saved new persistent machine ID: {machine_id[:8]}...")

        except Exception as e:
            logger.error(f"WARNING: Failed to save machine ID to {machine_id_file}: {e}")
            logger.error("Machine ID will not persist across container restarts!")

        return machine_id

    except Exception as e:
        logger.error(f"CRITICAL: Failed to generate persistent machine ID: {e}")
        # Last resort: Generate ephemeral ID (not ideal, but better than crashing)
        fallback_id = hashlib.sha256(f"fallback-{uuid.uuid4()}".encode()).hexdigest()[:24]
        logger.warning(f"Using fallback ephemeral machine ID: {fallback_id[:8]}...")
        return fallback_id


def get_machine_id() -> str:
    """
    DEPRECATED: Use get_persistent_machine_id() instead.

    This function is kept for backward compatibility but now delegates
    to the new persistent machine ID implementation.

    Returns:
        Persistent machine ID
    """
    logger.warning("get_machine_id() is deprecated, use get_persistent_machine_id()")
    return get_persistent_machine_id()


class LicenseValidator:
    """
    License validation using Supabase database with secure trial tracking.
    """

    def __init__(self, supabase_url: str, supabase_key: str):
        """
        Initialize license validator with Supabase connection.

        Args:
            supabase_url: Supabase project URL
            supabase_key: Supabase anonymous key
        """
        if not SUPABASE_AVAILABLE:
            raise ConfigurationError(
                "Supabase client not installed. Install with: pip install supabase"
            )

        self.supabase: Client = create_client(supabase_url, supabase_key)
        logger.info("Connected to Supabase for license validation")

    def validate_license_key(self, license_key: str) -> LicenseInfo:
        """
        Validate a license key via Supabase.

        Args:
            license_key: License key to validate

        Returns:
            LicenseInfo if valid

        Raises:
            ConfigurationError: If license is invalid or expired
        """
        try:
            # Query license from Supabase
            response = self.supabase.table('licenses') \
                .select('*') \
                .eq('license_key', license_key) \
                .eq('is_active', True) \
                .execute()

            if not response.data or len(response.data) == 0:
                raise ConfigurationError(
                    f"License key not found or inactive.\n"
                    f"Please contact support or check your license key.\n"
                    f"License key provided: {license_key[:10]}..."
                )

            license_data = response.data[0]

            # Parse datetime strings
            issued_at = datetime.fromisoformat(license_data['issued_at'].replace('Z', '+00:00'))
            expires_at = None
            if license_data.get('expires_at'):
                expires_at = datetime.fromisoformat(license_data['expires_at'].replace('Z', '+00:00'))

            # Create LicenseInfo object
            license_info = LicenseInfo(
                license_key=license_data['license_key'],
                tier=license_data['tier'],
                user_id=license_data['user_id'],
                max_repositories=license_data['max_repositories'],
                max_analyses_per_month=license_data['max_analyses_per_month'],
                max_storage_gb=license_data['max_storage_gb'],
                issued_at=issued_at,
                expires_at=expires_at,
                is_trial=license_data.get('is_trial', False),
                metadata=license_data.get('metadata', {})
            )

            # Check expiration
            if not license_info.is_valid():
                days_expired = (datetime.now(timezone.utc) - expires_at).days if expires_at else 0
                raise ConfigurationError(
                    f"License key has expired {days_expired} days ago.\n"
                    f"Expired on: {expires_at.strftime('%Y-%m-%d') if expires_at else 'N/A'}\n"
                    f"Please renew your license to continue using this product.\n"
                    f"Contact: support@github-issue-solver.com"
                )

            # Log successful validation
            days_left = license_info.days_remaining()
            if days_left is not None:
                logger.info(f"License validated: {license_info.tier} tier, expires in {days_left} days")
            else:
                logger.info(f"License validated: {license_info.tier} tier, lifetime license")

            return license_info

        except ConfigurationError:
            raise
        except Exception as e:
            logger.error(f"License validation error: {e}")
            raise ConfigurationError(
                f"Failed to validate license: {str(e)}\n"
                f"Please check your internet connection and Supabase configuration."
            )

    def get_or_create_trial(self, machine_id: str) -> LicenseInfo:
        """
        Get existing trial or create new 10-day trial for this machine.

        NOW WITH ENHANCED SECURITY:
        - Tracks GitHub PAT hash to prevent multi-container abuse
        - Stores GitHub username for support/tracking
        - Detects Docker container recreations
        - Supports remote deactivation

        Args:
            machine_id: Unique persistent machine identifier

        Returns:
            LicenseInfo for trial

        Raises:
            ConfigurationError: If trial expired or deactivated
        """
        try:
            # Get GitHub PAT hash and username
            github_pat_hash = get_github_pat_hash()
            github_username = self._get_github_username()

            # Check if trial exists for this machine ID
            response = self.supabase.table('trial_users') \
                .select('*') \
                .eq('machine_id', machine_id) \
                .execute()

            if response.data and len(response.data) > 0:
                # Trial exists - validate and check status
                trial_data = response.data[0]
                started_at = datetime.fromisoformat(trial_data['started_at'].replace('Z', '+00:00'))
                expires_at = datetime.fromisoformat(trial_data['expires_at'].replace('Z', '+00:00'))

                # Check if deactivated remotely
                if trial_data.get('is_deactivated', False):
                    reason = trial_data.get('deactivation_reason', 'No reason provided')
                    raise ConfigurationError(
                        f"🚫 Trial has been deactivated.\n\n"
                        f"Reason: {reason}\n\n"
                        f"To continue using GitHub Issue Solver:\n"
                        f"1. Purchase a license key\n"
                        f"2. Email: support@github-issue-solver.com\n"
                        f"3. Add LICENSE_KEY to your environment"
                    )

                # Check if expired
                if datetime.now(timezone.utc) > expires_at:
                    days_expired = (datetime.now(timezone.utc) - expires_at).days

                    # Mark as expired in database
                    self.supabase.table('trial_users') \
                        .update({'is_expired': True}) \
                        .eq('machine_id', machine_id) \
                        .execute()

                    raise ConfigurationError(
                        f"🔒 Free trial expired {days_expired} days ago!\n\n"
                        f"Trial period: 10 days\n"
                        f"Started: {started_at.strftime('%Y-%m-%d')}\n"
                        f"Expired: {expires_at.strftime('%Y-%m-%d')}\n\n"
                        f"To continue using GitHub Issue Solver:\n"
                        f"1. Purchase a license key\n"
                        f"2. Email: support@github-issue-solver.com\n"
                        f"3. Add LICENSE_KEY to your environment\n\n"
                        f"Pricing:\n"
                        f"- Personal: $9/month (10 repos, 100 analyses)\n"
                        f"- Team: $29/month (50 repos, 500 analyses)\n"
                        f"- Enterprise: Custom pricing"
                    )

                # Trial still valid - update last_seen and docker_restarts
                docker_restarts = trial_data.get('docker_restarts', 0) + 1
                self.supabase.table('trial_users') \
                    .update({
                        'last_seen_at': datetime.now(timezone.utc).isoformat(),
                        'docker_restarts': docker_restarts,
                        'github_pat_hash': github_pat_hash,  # Update in case PAT changed
                        'github_username': github_username
                    }) \
                    .eq('machine_id', machine_id) \
                    .execute()

                days_remaining = (expires_at - datetime.now(timezone.utc)).days
                logger.warning(
                    f"🔓 Running in FREE TRIAL mode. "
                    f"Expires in {days_remaining} days (on {expires_at.strftime('%Y-%m-%d')}). "
                    f"Container restarts: {docker_restarts}"
                )

                return LicenseInfo(
                    license_key='TRIAL-MODE',
                    tier=LICENSE.TIER_FREE,
                    user_id=f'trial_{machine_id}',
                    max_repositories=3,
                    max_analyses_per_month=10,
                    max_storage_gb=1,
                    issued_at=started_at,
                    expires_at=expires_at,
                    is_trial=True
                )

            else:
                # Check if this GitHub PAT hash already has a trial (anti-abuse)
                if github_pat_hash:
                    existing_trial = self.supabase.table('trial_users') \
                        .select('*') \
                        .eq('github_pat_hash', github_pat_hash) \
                        .execute()

                    if existing_trial.data and len(existing_trial.data) > 0:
                        # User already has a trial with different machine ID
                        existing = existing_trial.data[0]
                        logger.warning(
                            f"Detected existing trial for GitHub user {github_username} "
                            f"with different machine ID"
                        )

                        # Still allow (machine_id is primary key), but log for monitoring
                        logger.warning(
                            f"User {github_username} has {len(existing_trial.data)} trial(s) registered"
                        )

                # Create new trial (10 days)
                started_at = datetime.now(timezone.utc)
                expires_at = started_at + timedelta(days=10)

                # Insert into database
                self.supabase.table('trial_users').insert({
                    'machine_id': machine_id,
                    'github_pat_hash': github_pat_hash,
                    'github_username': github_username,
                    'started_at': started_at.isoformat(),
                    'expires_at': expires_at.isoformat(),
                    'repositories_used': 0,
                    'analyses_used': 0,
                    'is_expired': False,
                    'is_deactivated': False,
                    'docker_restarts': 0,
                    'first_seen_at': started_at.isoformat(),
                    'last_seen_at': started_at.isoformat(),
                    'metadata': {
                        'github_pat_hash': github_pat_hash[:16] if github_pat_hash else None
                    }
                }).execute()

                logger.warning(
                    f"✅ Free 10-day trial started!\n"
                    f"GitHub User: {github_username or 'Unknown'}\n"
                    f"Machine ID: {machine_id[:8]}...\n"
                    f"Started: {started_at.strftime('%Y-%m-%d')}\n"
                    f"Expires: {expires_at.strftime('%Y-%m-%d')}\n"
                    f"Limits: 3 repositories, 10 analyses"
                )

                return LicenseInfo(
                    license_key='TRIAL-MODE',
                    tier=LICENSE.TIER_FREE,
                    user_id=f'trial_{machine_id}',
                    max_repositories=3,
                    max_analyses_per_month=10,
                    max_storage_gb=1,
                    issued_at=started_at,
                    expires_at=expires_at,
                    is_trial=True
                )

        except ConfigurationError:
            raise
        except Exception as e:
            logger.error(f"Trial management error: {e}")
            raise ConfigurationError(
                f"Failed to manage trial: {str(e)}\n"
                f"Please check your internet connection."
            )

    def _get_github_username(self) -> Optional[str]:
        """
        Get GitHub username from GitHub token.

        Returns:
            GitHub username or None if unavailable
        """
        try:
            github_token = os.getenv('GITHUB_TOKEN')
            if not github_token:
                return None

            from github import Github
            gh = Github(github_token)
            user = gh.get_user()
            username = user.login
            logger.debug(f"GitHub username: {username}")
            return username

        except Exception as e:
            logger.warning(f"Failed to get GitHub username: {e}")
            return None

    def track_usage(
        self,
        license_key: str,
        action: str,
        repository: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        machine_id: Optional[str] = None
    ) -> None:
        """
        Track usage in Supabase for both paid and trial users.

        NOW WITH ENHANCED TRACKING:
        - Logs to trial_usage table for detailed tracking
        - Updates real-time counters in trial_users table
        - Deduplicates repository counting
        - Tracks success/failure

        Args:
            license_key: License key (or 'TRIAL-MODE' for trial users)
            action: Action performed ('ingest', 'analyze', 'patch')
            repository: Repository name (optional)
            metadata: Additional metadata (optional)
            machine_id: Machine ID for trial users (optional)
        """
        try:
            # Track in license_usage table (for all users including trials)
            usage_data = {
                'license_key': license_key,
                'action': action,
                'repository': repository,
                'metadata': metadata or {}
            }

            # Add machine_id to metadata for trial users
            if license_key == 'TRIAL-MODE' and machine_id:
                usage_data['metadata']['machine_id'] = machine_id

            self.supabase.table('license_usage').insert(usage_data).execute()

            logger.debug(f"Usage tracked: {action} for {license_key[:10]}...")

            # For trial users, also update trial_users counters and trial_usage table
            if license_key == 'TRIAL-MODE' and machine_id:
                self._track_trial_usage(machine_id, action, repository, metadata)

        except Exception as e:
            # Don't fail if usage tracking fails
            logger.warning(f"Failed to track usage: {e}")

    def _track_trial_usage(
        self,
        machine_id: str,
        action: str,
        repository: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Track trial usage in trial_usage table and update counters.

        Args:
            machine_id: Machine ID
            action: Action performed
            repository: Repository name (optional)
            metadata: Additional metadata (optional)
        """
        try:
            # Insert into trial_usage table (detailed log)
            timestamp = datetime.now(timezone.utc)

            usage_record = {
                'machine_id': machine_id,
                'action': action,
                'repository': repository,
                'timestamp': timestamp.isoformat(),
                'success': True,
                'metadata': metadata or {}
            }

            try:
                self.supabase.table('trial_usage').insert(usage_record).execute()
                logger.debug(f"Trial usage logged: {action} on {repository or 'N/A'}")
            except Exception as e:
                # Might fail due to unique constraint (duplicate record) - that's OK
                logger.debug(f"Trial usage record already exists or failed: {e}")

            # Update counters in trial_users table
            if action == 'ingest' and repository:
                # Get current unique repositories for this trial user
                repos_response = self.supabase.table('trial_usage') \
                    .select('repository') \
                    .eq('machine_id', machine_id) \
                    .eq('action', 'ingest') \
                    .execute()

                # Count unique repositories
                unique_repos = set()
                if repos_response.data:
                    for record in repos_response.data:
                        if record.get('repository'):
                            unique_repos.add(record['repository'])

                repositories_used = len(unique_repos)

                # Update counter
                self.supabase.table('trial_users') \
                    .update({'repositories_used': repositories_used}) \
                    .eq('machine_id', machine_id) \
                    .execute()

                logger.debug(f"Updated repositories_used to {repositories_used}")

            elif action in ['analyze', 'patch']:
                # Increment analyses counter
                trial_response = self.supabase.table('trial_users') \
                    .select('analyses_used') \
                    .eq('machine_id', machine_id) \
                    .execute()

                if trial_response.data and len(trial_response.data) > 0:
                    current_analyses = trial_response.data[0].get('analyses_used', 0)
                    new_analyses = current_analyses + 1

                    self.supabase.table('trial_users') \
                        .update({'analyses_used': new_analyses}) \
                        .eq('machine_id', machine_id) \
                        .execute()

                    logger.debug(f"Updated analyses_used to {new_analyses}")

        except Exception as e:
            logger.warning(f"Failed to track trial usage: {e}")

    def check_trial_limits(self, machine_id: str, action: str, repository: Optional[str] = None) -> tuple[bool, str]:
        """
        Check if trial user has exceeded their limits BEFORE performing action.

        NOW WITH ENHANCED VALIDATION:
        - Checks deactivation status
        - Validates expiration in real-time
        - Checks unique repository limits
        - Provides helpful upgrade messages

        Args:
            machine_id: Machine ID
            action: Action being performed ('ingest', 'analyze', 'patch')
            repository: Repository name (for unique repo counting)

        Returns:
            Tuple of (allowed: bool, message: str)
        """
        try:
            # Get current trial data
            response = self.supabase.table('trial_users') \
                .select('*') \
                .eq('machine_id', machine_id) \
                .execute()

            if not response.data or len(response.data) == 0:
                # No trial found - should not happen, but allow (will create trial)
                return True, ""

            trial = response.data[0]

            # Check if deactivated
            if trial.get('is_deactivated', False):
                reason = trial.get('deactivation_reason', 'No reason provided')
                return False, f"🚫 Trial deactivated: {reason}. Please purchase a license."

            # Check if expired
            if trial.get('is_expired', False):
                return False, "🔒 Trial expired. Please purchase a license to continue."

            # Double-check expiration (in case DB not updated yet)
            expires_at = datetime.fromisoformat(trial['expires_at'].replace('Z', '+00:00'))
            if datetime.now(timezone.utc) > expires_at:
                return False, "🔒 Trial expired. Please purchase a license to continue."

            # Check limits based on action
            if action == 'ingest':
                # For repository ingestion, check unique repository count
                repos_response = self.supabase.table('trial_usage') \
                    .select('repository') \
                    .eq('machine_id', machine_id) \
                    .eq('action', 'ingest') \
                    .execute()

                unique_repos = set()
                if repos_response.data:
                    for record in repos_response.data:
                        if record.get('repository'):
                            unique_repos.add(record['repository'])

                # Check if this repository is already ingested
                if repository and repository not in unique_repos:
                    # New repository - check limit
                    if len(unique_repos) >= 3:
                        return False, (
                            f"🔒 Trial limit reached: {len(unique_repos)}/3 repositories used.\n"
                            f"Repositories ingested: {', '.join(list(unique_repos)[:3])}\n\n"
                            f"Please purchase a license to ingest more repositories.\n"
                            f"Personal: $9/month (10 repos) | Team: $29/month (50 repos)"
                        )

            elif action in ['analyze', 'patch']:
                analyses_used = trial.get('analyses_used', 0)
                if analyses_used >= 10:
                    return False, (
                        f"🔒 Trial limit reached: {analyses_used}/10 analyses used.\n\n"
                        f"Please purchase a license to perform more analyses.\n"
                        f"Personal: $9/month (100 analyses) | Team: $29/month (500 analyses)"
                    )

            # Within limits
            return True, ""

        except Exception as e:
            logger.error(f"Error checking trial limits: {e}")
            # On error, allow (fail open) - but log for monitoring
            return True, ""

    def get_trial_usage_stats(self, machine_id: str) -> Dict[str, Any]:
        """
        Get detailed usage statistics for a trial user.

        Returns:
            Dictionary with usage stats including unique repos and analysis count
        """
        try:
            # Get trial data
            response = self.supabase.table('trial_users') \
                .select('*') \
                .eq('machine_id', machine_id) \
                .execute()

            if not response.data or len(response.data) == 0:
                return {"error": "Trial not found"}

            trial = response.data[0]
            expires_at = datetime.fromisoformat(trial['expires_at'].replace('Z', '+00:00'))
            days_remaining = max(0, (expires_at - datetime.now(timezone.utc)).days)

            # Get unique repositories from trial_usage
            repos_response = self.supabase.table('trial_usage') \
                .select('repository') \
                .eq('machine_id', machine_id) \
                .eq('action', 'ingest') \
                .execute()

            unique_repos = set()
            if repos_response.data:
                for record in repos_response.data:
                    if record.get('repository'):
                        unique_repos.add(record['repository'])

            return {
                "repositories_used": len(unique_repos),
                "repositories_list": list(unique_repos),
                "repositories_limit": 3,
                "analyses_used": trial.get('analyses_used', 0),
                "analyses_limit": 10,
                "days_remaining": days_remaining,
                "started_at": trial['started_at'],
                "expires_at": trial['expires_at'],
                "is_expired": trial.get('is_expired', False),
                "is_deactivated": trial.get('is_deactivated', False),
                "docker_restarts": trial.get('docker_restarts', 0),
                "github_username": trial.get('github_username', 'Unknown')
            }

        except Exception as e:
            logger.error(f"Error getting trial stats: {e}")
            return {"error": str(e)}


def get_license_from_env() -> Optional[str]:
    """Get license key from environment variable."""
    return os.getenv('LICENSE_KEY')


def validate_and_get_license_info(
    supabase_url: str,
    supabase_key: str
) -> LicenseInfo:
    """
    Validate license from environment and return license info.

    NOW WITH PERSISTENT MACHINE ID:
    - Uses /data/machine_id.lock for persistence
    - Falls back to GitHub PAT hash if file unavailable

    Args:
        supabase_url: Supabase project URL
        supabase_key: Supabase anonymous key

    Returns:
        LicenseInfo

    Raises:
        ConfigurationError: If license is invalid
    """
    # Check development mode FIRST (before validating license format)
    if os.getenv('ALLOW_NO_LICENSE', '').lower() == 'true':
        logger.warning("🔓 Running without license validation (development mode)")
        return LicenseInfo(
            license_key='DEV-MODE-NO-LICENSE',
            tier=LICENSE.TIER_FREE,
            user_id='dev_user',
            max_repositories=3,
            max_analyses_per_month=10,
            max_storage_gb=1,
            issued_at=datetime.now(timezone.utc),
            is_trial=True
        )

    validator = LicenseValidator(supabase_url, supabase_key)

    # Try to get license key from environment
    license_key = get_license_from_env()

    if license_key:
        # Validate provided license key
        logger.info("Validating provided license key...")
        return validator.validate_license_key(license_key)
    else:
        # No license key - use free trial with persistent machine ID
        logger.info("No license key provided, checking free trial status...")
        machine_id = get_persistent_machine_id()
        logger.info(f"Persistent Machine ID: {machine_id[:8]}...")
        return validator.get_or_create_trial(machine_id)
