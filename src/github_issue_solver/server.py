"""
GitHub Issue Solver MCP Server

A professional Model Context Protocol server for automated GitHub issue resolution,
repository analysis, and intelligent patch generation using FastMCP.

This server follows MCP best practices with proper separation of concerns,
robust error handling, and comprehensive health monitoring.
"""

import asyncio
import sys
import traceback
from typing import Dict, Any
from datetime import datetime
from loguru import logger

from mcp.server.fastmcp import FastMCP

from .config import Config
from .exceptions import GitHubIssueSolverError, ConfigurationError
from .license import LicenseValidator
from .services import (
    StateManager,
    RepositoryService,
    EmbeddingService,
    IngestionService,
    AnalysisService,
    PatchService,
    HealthService,
    LearningService,
)
from .services.llm_service import LLMService

# Loguru logger is already configured in main.py


class GitHubIssueSolverServer:
    """Main MCP server class with proper service organization."""
    
    def __init__(self, env_file: str = None):
        """
        Initialize the GitHub Issue Solver MCP server.
        
        Args:
            env_file: Optional path to environment file
        """
        self.mcp = FastMCP("github-issue-resolver")
        self.config = None
        self.services = {}
        self._initialized = False
        self._health_monitoring_active = False
        
        # Initialize configuration
        try:
            self.config = Config(env_file)
            logger.info("✅ Configuration loaded successfully")
        except ConfigurationError as e:
            logger.error(f"❌ Configuration error: {e.message}")
            if hasattr(e, 'details') and 'missing_variables' in e.details:
                logger.error("Required environment variables:")
                for var in e.details['missing_variables']:
                    logger.error(f"  - {var}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"❌ Unexpected configuration error: {e}")
            sys.exit(1)
        
        # Initialize services
        self._initialize_services()
        
        # Register tools
        self._register_tools()
        
        # Mark as initialized
        self._initialized = True
        logger.info("🚀 GitHub Issue Solver MCP Server initialized successfully")
    
    def _initialize_services(self) -> None:
        """Initialize all services with proper dependency injection."""
        try:
            # License validator for usage tracking
            self.services['license_validator'] = LicenseValidator(
                self.config.supabase_url,
                self.config.supabase_anon_key
            )

            # Core services
            self.services['state_manager'] = StateManager(self.config)
            self.services['repository'] = RepositoryService(self.config)

            # Initialize EmbeddingService first
            self.services['embedding'] = EmbeddingService(self.config)

            # Initialize LLM service
            self.services['llm'] = LLMService(self.config)
            logger.info(f"LLM provider: {self.services['llm'].provider_name} ({self.services['llm'].model_name})")

            # Business logic services
            self.services['ingestion'] = IngestionService(
                self.config,
                self.services['repository'],
                self.services['state_manager'],
                self.services['embedding']
            )

            self.services['analysis'] = AnalysisService(
                self.config,
                self.services['repository'],
                self.services['state_manager'],
                self.services['llm']
            )

            self.services['patch'] = PatchService(
                self.config,
                self.services['state_manager'],
                self.services['llm']
            )
            
            self.services['health'] = HealthService(
                self.config,
                self.services['state_manager'],
                self.services['repository']
            )

            self.services['learning'] = LearningService(
                self.config,
                self.services['embedding']
            )
            
            logger.info("✅ All services initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Service initialization failed: {e}")
            raise GitHubIssueSolverError(f"Failed to initialize services: {str(e)}", cause=e)
    
    def _register_tools(self) -> None:
        """Register all MCP tools with the server."""
        try:
            # Ingestion tools
            self._register_ingestion_tools()
            
            # Analysis tools  
            self._register_analysis_tools()
            
            # Patch generation tools
            self._register_patch_tools()
            
            # Management and health tools
            self._register_management_tools()

            # Learning tools
            self._register_learning_tools()

            logger.info("✅ All MCP tools registered successfully")
            
        except Exception as e:
            logger.error(f"❌ Tool registration failed: {e}")
            raise GitHubIssueSolverError(f"Failed to register tools: {str(e)}", cause=e)
    
    def _register_ingestion_tools(self) -> None:
        """Register repository ingestion tools."""
        ingestion_service = self.services['ingestion']
        
        @self.mcp.tool()
        async def start_repository_ingestion(repo_name: str) -> str:
            """Start the multi-step repository ingestion process."""
            try:
                # CHECK TRIAL LIMITS BEFORE ALLOWING INGESTION
                if self.config.license_info and self.config.license_info.is_trial:
                    license_validator = self.services.get('license_validator')
                    if license_validator and self.config.machine_id:
                        allowed, message = license_validator.check_trial_limits(
                            self.config.machine_id,
                            'ingest'
                        )
                        if not allowed:
                            return f"❌ {message}"

                result = await ingestion_service.start_repository_ingestion(repo_name)
                if result.success:
                    return self._format_ingestion_start_success(repo_name, result)
                else:
                    return f"❌ **Ingestion Start Failed**: {result.error_message}"
            except Exception as e:
                logger.error(f"Error in start_repository_ingestion: {e}")
                return f"❌ **Error**: {str(e)}"
        
        @self.mcp.tool()
        async def ingest_repository_docs(repo_name: str) -> str:
            """Ingest documentation (Step 1 of 4)."""
            return await self._execute_ingestion_step("docs", repo_name, ingestion_service.ingest_documentation)
        
        @self.mcp.tool()
        async def ingest_repository_code(repo_name: str) -> str:
            """Ingest source code (Step 2 of 4)."""
            return await self._execute_ingestion_step("code", repo_name, ingestion_service.ingest_code)
        
        @self.mcp.tool()
        async def ingest_repository_issues(repo_name: str, max_issues: int = 100) -> str:
            """Ingest issues history (Step 3 of 4)."""
            return await self._execute_ingestion_step("issues", repo_name, 
                                                    lambda r: ingestion_service.ingest_issues(r, max_issues))
        
        @self.mcp.tool()
        async def ingest_repository_prs(repo_name: str, max_prs: int = 15) -> str:
            """Ingest PR history (Step 4 of 4). Default is 15 PRs to prevent timeouts on repos with many PRs."""
            return await self._execute_ingestion_step("prs", repo_name, 
                                                    lambda r: ingestion_service.ingest_prs(r, max_prs), 
                                                    is_final_step=True)
    
    def _register_analysis_tools(self) -> None:
        """Register analysis tools."""
        analysis_service = self.services['analysis']
        state_manager = self.services['state_manager']
        repository_service = self.services['repository']
        ingestion_service = self.services['ingestion']  # FIX: Was missing, causing 'ingestion_service' undefined error
        
        @self.mcp.tool()
        async def analyze_github_issue_tool(issue_url: str) -> dict:
            """Analyze a GitHub issue using AI and repository knowledge base."""
            try:
                # CHECK TRIAL LIMITS BEFORE ALLOWING ANALYSIS
                if self.config.license_info and self.config.license_info.is_trial:
                    license_validator = self.services.get('license_validator')
                    if license_validator and self.config.machine_id:
                        allowed, message = license_validator.check_trial_limits(
                            self.config.machine_id,
                            'analyze'
                        )
                        if not allowed:
                            return {"success": False, "error": message}

                result = await analysis_service.analyze_github_issue(issue_url)

                # Track usage for trial users after successful analysis
                if result.success and self.config.license_info and self.config.license_info.is_trial:
                    try:
                        license_validator = self.services.get('license_validator')
                        if license_validator and self.config.machine_id:
                            license_validator.track_usage(
                                license_key=self.config.license_info.license_key,
                                action='analyze',
                                repository=result.metadata.get('repository', 'unknown'),
                                metadata={
                                    'issue_url': issue_url,
                                    'completed_at': datetime.now().isoformat()
                                },
                                machine_id=self.config.machine_id
                            )
                            logger.info(f"✅ Trial usage tracked for analysis")
                    except Exception as usage_error:
                        logger.warning(f"Failed to track trial usage: {usage_error}")

                return result.to_dict()
            except Exception as e:
                logger.error(f"Error in analyze_github_issue_tool: {e}")
                return {
                    "success": False,
                    "error": str(e)
                }
        
        @self.mcp.tool()
        async def get_repository_status(repo_name: str) -> str:
            """Get detailed repository ingestion status."""
            try:
                progress = await ingestion_service.get_ingestion_progress(repo_name)
                return self._format_repository_status(progress)
            except Exception as e:
                logger.error(f"Error in get_repository_status: {e}")
                return f"❌ **Error**: {str(e)}"
        
        @self.mcp.tool()
        async def get_repository_info(repo_name: str) -> str:
            """Get GitHub repository information."""
            try:
                info = await repository_service.get_repository_info(repo_name)
                return self._format_repository_info(info)
            except Exception as e:
                logger.error(f"Error in get_repository_info: {e}")
                return f"❌ **Error**: {str(e)}"
        
        @self.mcp.tool()
        async def validate_repository_tool(repo_name: str) -> str:
            """Validate repository access and permissions."""
            try:
                is_valid = await repository_service.validate_repository(repo_name)
                if is_valid:
                    info = await repository_service.get_repository_info(repo_name)
                    return self._format_repository_validation_success(repo_name, info)
                else:
                    return f"❌ **Repository Validation Failed**: Repository '{repo_name}' is not accessible"
            except Exception as e:
                logger.error(f"Error in validate_repository_tool: {e}")
                return f"❌ **Error**: {str(e)}"
        
        @self.mcp.tool()
        async def list_ingested_repositories() -> str:
            """List all ingested repositories."""
            try:
                repositories = state_manager.list_repositories()
                if not repositories:
                    return self._format_no_repositories()
                
                repo_details = []
                for repo_name in repositories:
                    status = state_manager.get_repository_status(repo_name)
                    repo_details.append({
                        'name': repo_name,
                        'status': status.overall_status.value if status else 'unknown',
                        'documents': status.total_documents if status else 0,
                        'updated': status.updated_at.strftime('%Y-%m-%d %H:%M:%S') if status and status.updated_at else 'unknown'
                    })
                
                return self._format_repository_list(repo_details)
            except Exception as e:
                logger.error(f"Error in list_ingested_repositories: {e}")
                return f"❌ **Error**: {str(e)}"
    
    def _register_patch_tools(self) -> None:
        """Register patch generation tools."""
        patch_service = self.services['patch']
        
        @self.mcp.tool()
        async def generate_code_patch_tool(issue_body: str, repo_full_name: str) -> dict:
            """Generate code patches for issue resolution."""
            try:
                # CHECK TRIAL LIMITS BEFORE ALLOWING PATCH GENERATION
                if self.config.license_info and self.config.license_info.is_trial:
                    license_validator = self.services.get('license_validator')
                    if license_validator and self.config.machine_id:
                        allowed, message = license_validator.check_trial_limits(
                            self.config.machine_id,
                            'patch'
                        )
                        if not allowed:
                            return {"success": False, "error": message}

                result = await patch_service.generate_code_patch(issue_body, repo_full_name)

                # Track usage for trial users after successful patch generation
                if result.success and self.config.license_info and self.config.license_info.is_trial:
                    try:
                        license_validator = self.services.get('license_validator')
                        if license_validator and self.config.machine_id:
                            license_validator.track_usage(
                                license_key=self.config.license_info.license_key,
                                action='patch',
                                repository=repo_full_name,
                                metadata={
                                    'completed_at': datetime.now().isoformat()
                                },
                                machine_id=self.config.machine_id
                            )
                            logger.info(f"✅ Trial usage tracked for patch generation")
                    except Exception as usage_error:
                        logger.warning(f"Failed to track trial usage: {usage_error}")

                return result.to_dict()
            except Exception as e:
                logger.error(f"Error in generate_code_patch_tool: {e}")
                return {
                    "success": False,
                    "error": str(e)
                }
        
        @self.mcp.tool()
        async def get_patch_guidance(repo_name: str, issue_description: str) -> str:
            """Get implementation guidance for manual issue resolution."""
            try:
                guidance = await patch_service.get_implementation_guidance(repo_name, issue_description)
                return self._format_patch_guidance(guidance)
            except Exception as e:
                logger.error(f"Error in get_patch_guidance: {e}")
                return f"❌ **Error**: {str(e)}"
        
        @self.mcp.tool()
        async def get_repository_structure(repo_name: str, max_files: int = 50) -> str:
            """Get repository structure from ingested data."""
            try:
                structure = await self.services['analysis'].get_repository_structure_summary(repo_name)
                return self._format_repository_structure(structure, max_files)
            except Exception as e:
                logger.error(f"Error in get_repository_structure: {e}")
                return f"❌ **Error**: {str(e)}"
    
    def _register_management_tools(self) -> None:
        """Register management and health tools."""
        health_service = self.services['health']
        state_manager = self.services['state_manager']
        
        @self.mcp.tool()
        async def get_health_status_tool() -> str:
            """Get comprehensive health status of the MCP server."""
            try:
                health = await health_service.get_health_status()
                summary = await health_service.get_health_summary()
                return self._format_health_status(health, summary)
            except Exception as e:
                logger.error(f"Error in get_health_status_tool: {e}")
                return f"❌ **Health Check Error**: {str(e)}"
        
        @self.mcp.tool()
        async def clear_repository_data_tool(repo_name: str, confirm: bool = False) -> str:
            """Clear all data for a specific repository."""
            try:
                if not confirm:
                    return self._format_clear_confirmation(repo_name)
                
                # Delete from state manager
                deleted = state_manager.delete_repository_status(repo_name)
                
                if deleted:
                    return f"✅ **Repository Data Cleared**: All data for '{repo_name}' has been removed."
                else:
                    return f"❌ **Repository Not Found**: No data found for '{repo_name}'."
                    
            except Exception as e:
                logger.error(f"Error in clear_repository_data_tool: {e}")
                return f"❌ **Clear Operation Failed**: {str(e)}"
        
        @self.mcp.tool()
        async def cleanup_old_data_tool(max_age_days: int = 30) -> str:
            """Cleanup old repository data to free resources."""
            try:
                result = await health_service.cleanup_old_data(max_age_days)
                return self._format_cleanup_result(result)
            except Exception as e:
                logger.error(f"Error in cleanup_old_data_tool: {e}")
                return f"❌ **Cleanup Failed**: {str(e)}"

        @self.mcp.tool()
        async def get_trial_usage() -> str:
            """Get current trial usage statistics."""
            try:
                if not self.config.license_info or not self.config.license_info.is_trial:
                    return "ℹ️ **This command is only available for trial users.**\n\nYou have a paid license. No usage limits apply."

                license_validator = self.services.get('license_validator')
                if not license_validator or not self.config.machine_id:
                    return "❌ **Error**: Cannot retrieve trial statistics."

                stats = license_validator.get_trial_usage_stats(self.config.machine_id)

                if 'error' in stats:
                    return f"❌ **Error**: {stats['error']}"

                days_remaining = stats['days_remaining']
                repos_used = stats['repositories_used']
                repos_limit = stats['repositories_limit']
                analyses_used = stats['analyses_used']
                analyses_limit = stats['analyses_limit']
                is_expired = stats['is_expired']

                status_icon = "🔒" if is_expired else "✅"
                status_text = "**Trial Expired!**" if is_expired else "**Trial Active**"

                return f"""📊 **Trial Usage Statistics**

{status_icon} {status_text}

🗂️ **Repositories**: {repos_used}/{repos_limit} used
   {'⚠️ Limit reached!' if repos_used >= repos_limit else f'• {repos_limit - repos_used} remaining'}

📝 **Analyses**: {analyses_used}/{analyses_limit} used
   {'⚠️ Limit reached!' if analyses_used >= analyses_limit else f'• {analyses_limit - analyses_used} remaining'}

⏰ **Trial Period**:
   • Days Remaining: {days_remaining} days
   • Started: {stats['started_at'][:10]}
   • Expires: {stats['expires_at'][:10]}

💡 **Upgrade to unlock unlimited access:**
   • Personal: $9/month (10 repos, 100 analyses)
   • Team: $29/month (50 repos, 500 analyses)
   • Enterprise: Custom pricing (unlimited)

📧 **Contact**: support@yourcompany.com for license purchase"""
            except Exception as e:
                logger.error(f"Error getting trial stats: {e}")
                return f"❌ **Error**: {str(e)}"

    def _register_learning_tools(self) -> None:
        """Register learning management tools."""
        learning_service = self.services['learning']

        @self.mcp.tool()
        async def add_pr_learning(
            repo_name: str,
            learning_type: str,
            content: dict
        ) -> dict:
            """
            Add a new learning to the repository's knowledge base.

            Args:
                repo_name: Repository name (e.g., 'owner/repo')
                learning_type: Type of learning - one of:
                    - 'never_do': Rule for things that should never be done
                    - 'pattern': Code pattern with do/don't examples
                    - 'checklist': Item to add to a checklist
                    - 'pr_takeaway': Lessons learned from a PR
                content: Learning content (structure depends on type):
                    For 'never_do': {"rule": "...", "reason": "...", "source_pr": "..."}
                    For 'pattern': {"name": "...", "language": "...", "description": "...",
                                   "do_example": "...", "dont_example": "...", "tags": [...]}
                    For 'checklist': {"checklist_name": "...", "item": "..."}
                    For 'pr_takeaway': {"original_pr": "...", "title": "...", "result": "...",
                                       "lessons": [{"lesson": "...", "action": "..."}]}

            Returns:
                dict with success status and created learning
            """
            try:
                if learning_type == "never_do":
                    result = learning_service.add_never_do_rule(
                        repo_name=repo_name,
                        rule=content.get("rule", ""),
                        reason=content.get("reason", ""),
                        source_pr=content.get("source_pr")
                    )
                    return {"success": True, "learning": result.to_dict(), "type": "never_do"}

                elif learning_type == "pattern":
                    result = learning_service.add_pattern(
                        repo_name=repo_name,
                        name=content.get("name", ""),
                        language=content.get("language", "general"),
                        description=content.get("description", ""),
                        do_example=content.get("do_example", ""),
                        dont_example=content.get("dont_example", ""),
                        tags=content.get("tags", []),
                        source_pr=content.get("source_pr")
                    )
                    return {"success": True, "learning": result.to_dict(), "type": "pattern"}

                elif learning_type == "checklist":
                    learning_service.add_checklist_item(
                        repo_name=repo_name,
                        checklist_name=content.get("checklist_name", "general"),
                        item=content.get("item", "")
                    )
                    return {"success": True, "type": "checklist", "message": "Checklist item added"}

                elif learning_type == "pr_takeaway":
                    result = learning_service.add_pr_takeaway(
                        repo_name=repo_name,
                        original_pr=content.get("original_pr", ""),
                        title=content.get("title", ""),
                        result=content.get("result", ""),
                        lessons=content.get("lessons", []),
                        merged_pr=content.get("merged_pr"),
                        files_touched=content.get("files_touched", [])
                    )
                    return {"success": True, "learning": result.to_dict(), "type": "pr_takeaway"}

                else:
                    return {"success": False, "error": f"Unknown learning type: {learning_type}"}

            except Exception as e:
                logger.error(f"Error adding learning: {e}")
                return {"success": False, "error": str(e)}

        @self.mcp.tool()
        async def get_repository_learnings(
            repo_name: str,
            include_patterns: bool = True,
            include_checklists: bool = True
        ) -> dict:
            """
            Get all learnings for a repository.

            Args:
                repo_name: Repository name (e.g., 'owner/repo')
                include_patterns: Include code patterns in response
                include_checklists: Include checklists in response

            Returns:
                dict with all learnings for the repository
            """
            try:
                learnings = learning_service.load_learnings(repo_name)
                result = {
                    "success": True,
                    "repository": repo_name,
                    "total_learnings": learnings.get_total_learnings_count(),
                    "never_do": [rule.to_dict() for rule in learnings.never_do],
                    "pr_takeaways": [t.to_dict() for t in learnings.pr_takeaways],
                }

                if include_patterns:
                    result["patterns"] = [p.to_dict() for p in learnings.patterns]

                if include_checklists:
                    result["checklists"] = learnings.checklists
                    result["file_groups"] = learnings.file_groups

                return result

            except Exception as e:
                logger.error(f"Error getting learnings: {e}")
                return {"success": False, "error": str(e)}

        @self.mcp.tool()
        async def search_similar_learnings(
            repo_name: str,
            query: str,
            n_results: int = 5
        ) -> dict:
            """
            Search learnings using semantic similarity.

            Args:
                repo_name: Repository name (e.g., 'owner/repo')
                query: Natural language search query (e.g., 'how to handle recursion')
                n_results: Number of results to return (default 5)

            Returns:
                dict with matching learnings sorted by relevance
            """
            try:
                results = learning_service.search_learnings(
                    repo_name=repo_name,
                    query=query,
                    n_results=n_results
                )
                return {
                    "success": True,
                    "repository": repo_name,
                    "query": query,
                    "results": results,
                    "count": len(results)
                }

            except Exception as e:
                logger.error(f"Error searching learnings: {e}")
                return {"success": False, "error": str(e)}

        @self.mcp.tool()
        async def get_pre_pr_checklist(
            repo_name: str,
            files_changed: list = None
        ) -> dict:
            """
            Get relevant checklist and warnings before making a PR.

            Call this tool BEFORE finalizing any PR to get:
            - Never-do warnings (critical rules to avoid)
            - Applicable checklists based on files changed
            - Relevant code patterns

            Args:
                repo_name: Repository name (e.g., 'owner/repo')
                files_changed: Optional list of files being modified

            Returns:
                dict with warnings, checklists, and patterns to review
            """
            try:
                result = learning_service.get_pre_pr_checklist(
                    repo_name=repo_name,
                    files_changed=files_changed
                )
                result["success"] = True
                return result

            except Exception as e:
                logger.error(f"Error getting pre-PR checklist: {e}")
                return {"success": False, "error": str(e)}

        @self.mcp.tool()
        async def import_learnings_from_markdown(
            repo_name: str,
            markdown_path: str
        ) -> dict:
            """
            Import learnings from an existing markdown file.

            Parses markdown files with the following structure:
            - ## NEVER DO section with numbered rules
            - ## Pre-PR Checklist section with ### subsections
            - ## Learned Patterns section with code examples
            - ## PR Takeaways section with lesson tables

            Args:
                repo_name: Repository name to store learnings under
                markdown_path: Full path to the markdown file

            Returns:
                dict with counts of imported items
            """
            try:
                counts = learning_service.import_from_markdown(
                    repo_name=repo_name,
                    markdown_path=markdown_path
                )
                return {
                    "success": True,
                    "repository": repo_name,
                    "imported": counts,
                    "total": sum(counts.values()),
                    "message": f"Successfully imported {sum(counts.values())} learnings"
                }

            except Exception as e:
                logger.error(f"Error importing learnings: {e}")
                return {"success": False, "error": str(e)}

    async def _execute_ingestion_step(self, step_name: str, repo_name: str, step_function, is_final_step: bool = False) -> str:
        """Execute an ingestion step with consistent error handling."""
        try:
            result = await step_function(repo_name)
            if result.success:
                # Track license usage when final step completes
                if is_final_step:
                    try:
                        license_validator = self.services.get('license_validator')
                        if license_validator and self.config.license_info:
                            license_validator.track_usage(
                                license_key=self.config.license_info.license_key,
                                action='ingest',
                                repository=repo_name,
                                metadata={
                                    'completed_at': datetime.now().isoformat(),
                                    'final_step': step_name
                                },
                                machine_id=self.config.machine_id if self.config.license_info.is_trial else None
                            )
                            logger.info(f"✅ License usage tracked for repository: {repo_name}")
                    except Exception as usage_error:
                        # Don't fail the ingestion if usage tracking fails
                        logger.warning(f"Failed to track license usage: {usage_error}")

                progress = await self.services['ingestion'].get_ingestion_progress(repo_name)
                return self._format_ingestion_step_success(step_name, result, progress, is_final_step)
            else:
                return f"❌ **Step Failed**: {result.error_message}"
        except Exception as e:
            logger.error(f"Error in {step_name} ingestion: {e}")
            return f"❌ **Step Failed**: {str(e)}"
    
    def _format_ingestion_start_success(self, repo_name: str, result) -> str:
        """Format successful ingestion start message."""
        return f"""🚀 **Repository Ingestion Started Successfully!**

📂 **Repository**: {repo_name}
🎯 **Status**: Initialization Complete - Ready for Step-by-Step Ingestion

📋 **4-Step Ingestion Plan:**

**Step 1: Documentation Ingestion** 📚
• Command: `ingest_repository_docs('{repo_name}')`
• Fetches and embeds README files, wikis, and documentation

**Step 2: Code Analysis** 💻  
• Command: `ingest_repository_code('{repo_name}')`
• Analyzes and chunks source code files for context

**Step 3: Issues History** 🐛
• Command: `ingest_repository_issues('{repo_name}')`
• Processes recent issues for pattern recognition

**Step 4: PR History** 🔄
• Command: `ingest_repository_prs('{repo_name}')`
• Analyzes pull request history for solution patterns

💡 **Next Step:** Begin with Step 1:
`ingest_repository_docs('{repo_name}')`

✅ **Repository validated and ready for ingestion!**"""
    
    def _format_ingestion_step_success(self, step_name: str, result, progress: Dict[str, Any], is_final_step: bool) -> str:
        """Format successful ingestion step message."""
        step_names = {"docs": "Documentation", "code": "Code", "issues": "Issues", "prs": "PRs"}
        step_display = step_names.get(step_name, step_name.title())
        
        if is_final_step:
            return f"""🎉 **INGESTION COMPLETE! All 4 Steps Finished!**

🔄 **Step 4 Results - PR History:**
• Repository: {result.repo_name}
• Documents Stored: {result.documents_stored:,} chunks
• Processing Time: {result.duration_seconds:.2f}s

📊 **Final Summary:** {progress.get('total_documents', 0):,} total documents
🚀 **Ready for AI-powered issue resolution!**"""
        else:
            completion = progress.get('completion_percentage', 0)
            next_step_map = {"docs": "ingest_repository_code", "code": "ingest_repository_issues", "issues": "ingest_repository_prs"}
            next_step = next_step_map.get(step_name, "")
            
            return f"""✅ **Step Complete: {step_display} Ingested!**

📊 **Results:**
• Documents Stored: {result.documents_stored:,} chunks
• Processing Time: {result.duration_seconds:.2f}s
• Progress: {completion:.1f}% complete

🎯 **Next Step:** `{next_step}('{result.repo_name}')`"""
    
    def _format_repository_status(self, progress: Dict[str, Any]) -> str:
        """Format repository status message."""
        if not progress.get('initialized'):
            return f"❌ **Repository Not Found**: {progress.get('error', 'Unknown error')}"
        
        repo_name = progress.get('repo_name', 'Unknown')
        status = progress.get('overall_status', 'unknown')
        completion = progress.get('completion_percentage', 0)
        
        return f"""📊 **Repository Status: {repo_name}**

🎯 **Status**: {status.title()} ({completion:.1f}% complete)
📈 **Total Documents**: {progress.get('total_documents', 0):,}
📁 **Collections**: {len(progress.get('collections', []))}

**Step Progress:**
{self._format_step_progress(progress.get('steps', {}))}

💡 **Next Step:** {(progress.get('next_step') or 'Complete').replace('_', ' ').title()}"""
    
    def _format_step_progress(self, steps: Dict[str, Any]) -> str:
        """Format step progress details."""
        step_names = {"documentation": "📚 Docs", "code": "💻 Code", "issues": "🐛 Issues", "prs": "🔄 PRs"}
        lines = []
        
        for step, details in steps.items():
            name = step_names.get(step, step.title())
            status = details.get('status', 'pending')
            count = details.get('documents_stored', 0)
            
            if status == 'completed':
                lines.append(f"• {name}: ✅ {count:,} documents")
            elif status == 'in_progress':
                lines.append(f"• {name}: 🔄 In progress...")
            elif status == 'error':
                lines.append(f"• {name}: ❌ Error")
            else:
                lines.append(f"• {name}: ⏳ Pending")
        
        return '\n'.join(lines)
    
    
    def run(self, transport: str = 'stdio') -> None:
        """
        Run the MCP server with auto-recovery for non-fatal errors.

        Args:
            transport: Transport type ('stdio' or 'sse')
        """
        max_restarts = 3
        restart_count = 0

        while restart_count <= max_restarts:
            try:
                if not self._initialized:
                    raise GitHubIssueSolverError("Server not properly initialized")

                if restart_count == 0:
                    logger.info("🛠️ Available tools:")
                    logger.info("  🚀 Multi-Step Ingestion Tools:")
                    logger.info("    • start_repository_ingestion - Initialize repo and start ingestion")
                    logger.info("    • ingest_repository_docs - Step 1: Documentation")
                    logger.info("    • ingest_repository_code - Step 2: Source code")
                    logger.info("    • ingest_repository_issues - Step 3: Issues history")
                    logger.info("    • ingest_repository_prs - Step 4: PR history")
                    logger.info("  📊 Analysis & Patching Tools:")
                    logger.info("    • analyze_github_issue_tool - AI-powered issue analysis")
                    logger.info("    • generate_code_patch_tool - Generate solution patches")
                    logger.info("  📚 Learning Tools:")
                    logger.info("    • add_pr_learning - Add new learning to knowledge base")
                    logger.info("    • get_repository_learnings - Get all learnings for a repo")
                    logger.info("    • search_similar_learnings - Semantic search through learnings")
                    logger.info("    • get_pre_pr_checklist - Get checklist before making PR")
                    logger.info("    • import_learnings_from_markdown - Import from markdown file")
                    logger.info("  📋 Management Tools:")
                    logger.info("    • get_repository_status - Check ingestion progress")
                    logger.info("    • get_health_status_tool - Server health monitoring")

                    logger.info(f"🎯 Starting MCP server with {transport} transport...")

                    # Start health monitoring in background thread
                    self.services['health'].start_monitoring()
                    self._health_monitoring_active = True
                    logger.info("✅ Health monitoring background thread started")
                else:
                    logger.info(f"🔄 Restarting MCP server (attempt {restart_count}/{max_restarts})...")

                # Run the FastMCP server (this blocks until server stops)
                self.mcp.run(transport=transport)

                # Normal exit from run() means server stopped cleanly
                break

            except KeyboardInterrupt:
                logger.info("🛑 Server stopped by user")
                break
            except Exception as e:
                error_str = str(e)
                error_type = type(e).__name__

                # Check for recoverable MCP errors (client cancellation/timeout)
                is_recoverable = any([
                    "ClientNotification" in error_str,
                    "cancelled" in error_str.lower(),
                    "ValidationError" in error_type and "notifications" in error_str.lower(),
                    error_type == "ValidationError" and restart_count < max_restarts,
                ])

                if is_recoverable:
                    restart_count += 1
                    logger.warning(f"⚠️ Recoverable MCP error: {error_type}")
                    logger.info("💡 This typically happens when client times out or cancels a request.")

                    if restart_count <= max_restarts:
                        logger.info(f"🔄 Attempting auto-recovery ({restart_count}/{max_restarts})...")
                        import time
                        time.sleep(1)  # Brief pause before restart
                        continue
                    else:
                        logger.error(f"❌ Max restart attempts ({max_restarts}) exceeded. Shutting down.")
                        sys.exit(1)
                else:
                    logger.error(f"❌ Fatal server error: {e}")
                    logger.error(traceback.format_exc())
                    sys.exit(1)
            finally:
                if self._health_monitoring_active:
                    self.services['health'].stop_monitoring()
                    self._health_monitoring_active = False
                    logger.info("Health monitoring stopped")
    
    # Additional formatting methods would go here...
    def _format_repository_info(self, info: Dict[str, Any]) -> str:
        """Format repository information."""
        return f"""📋 **Repository Information**

✅ **Status**: Accessible
📂 **Name**: {info.get('full_name', 'Unknown')}
📝 **Description**: {info.get('description', 'No description')}
🌟 **Stars**: {info.get('stars', 0):,}
🍴 **Forks**: {info.get('forks', 0):,}
🐛 **Open Issues**: {info.get('open_issues', 0):,}
🌿 **Default Branch**: {info.get('default_branch', 'main')}
💻 **Language**: {info.get('language', 'Multiple/Unknown')}
🔒 **Visibility**: {'Private' if info.get('is_private') else 'Public'}
📅 **Last Updated**: {info.get('last_updated', 'Unknown')}

🎯 **Ready for ingestion and analysis!**"""
    
    def _format_repository_validation_success(self, repo_name: str, info: Dict[str, Any]) -> str:
        """Format successful repository validation."""
        return f"""✅ **Repository Validation Successful**

📂 **Repository**: {repo_name}
🌟 **Stars**: {info.get('stars', 0):,}
🍴 **Forks**: {info.get('forks', 0):,}
💻 **Language**: {info.get('language', 'Multiple/Unknown')}
🔒 **Access**: Repository is accessible for ingestion

🎯 **Next Step**: Run `start_repository_ingestion('{repo_name}')` to begin the 4-step ingestion process."""
    
    def _format_no_repositories(self) -> str:
        """Format message when no repositories are ingested."""
        return """📋 **No Repositories Ingested**

No repositories have been ingested into the knowledge base yet.

🎯 **To get started:**
1. Use `start_repository_ingestion('owner/repo')` to begin ingestion
2. Follow the 4-step process for complete ingestion
3. Then analyze issues and generate patches

💡 **Example**: `start_repository_ingestion('microsoft/vscode')`"""
    
    def _format_repository_list(self, repos: list) -> str:
        """Format list of ingested repositories."""
        lines = [f"📋 **Ingested Repositories ({len(repos)})**\n"]
        
        for repo in repos:
            status_icon = "✅" if repo['status'] == 'completed' else "🔄" if repo['status'] == 'in_progress' else "❌"
            lines.append(f"**{repo['name']}**")
            lines.append(f"• Status: {status_icon} {repo['status'].title()}")
            lines.append(f"• Documents: {repo['documents']:,}")
            lines.append(f"• Updated: {repo['updated']}\n")
        
        lines.append("🔧 **Available Operations:**")
        lines.append("• `analyze_github_issue_tool` - Analyze specific issues")
        lines.append("• `generate_code_patch_tool` - Generate solution patches")
        lines.append("• `get_repository_status` - Check detailed progress")
        
        return '\n'.join(lines)
    
    def _format_patch_guidance(self, guidance: Dict[str, Any]) -> str:
        """Format patch implementation guidance."""
        if 'error' in guidance:
            return f"❌ **Error**: {guidance['error']}"
        
        issue_type = guidance.get('issue_type', 'general').replace('_', ' ').title()
        
        return f"""🧭 **Implementation Guidance**

📋 **Issue Type**: {issue_type}
📂 **Repository**: {guidance.get('repository', 'Unknown')}

💡 **General Approach:**
{guidance.get('general_approach', 'Analyze the codebase and implement changes carefully.')}

🎯 **Specific Suggestions:**
{chr(10).join([f"• {suggestion}" for suggestion in guidance.get('specific_suggestions', [])])}

🔧 **Next Steps:**
{chr(10).join([f"{i+1}. {step}" for i, step in enumerate(guidance.get('next_steps', []))])}

📊 **Repository Context:**
• Total Documents: {guidance.get('repository_context', {}).get('total_documents', 0):,}
• Collections: {len(guidance.get('repository_context', {}).get('collections', []))}
"""
    
    def _format_repository_structure(self, structure: Dict[str, Any], max_files: int) -> str:
        """Format repository structure information."""
        if not structure.get('available'):
            return f"❌ **Structure Not Available**: {structure.get('error', 'Repository not ingested')}"
        
        repo_info = structure.get('repo_info', {})
        
        return f"""📂 **Repository Structure**

✅ **Status**: Available for analysis
📊 **Total Documents**: {structure.get('total_documents', 0):,}
📁 **Collections**: {len(structure.get('collections', []))}
🔄 **Ingestion Status**: {structure.get('ingestion_status', 'unknown').title()}

📝 **Repository Info:**
• Language: {repo_info.get('language', 'Multiple/Unknown')}
• Size: {repo_info.get('size_kb', 0)} KB
• Default Branch: {repo_info.get('default_branch', 'main')}

📁 **Available Collections:**
{chr(10).join([f"  • {col}" for col in structure.get('collections', [])])}

🔧 **Ready for:**
• Issue analysis with `analyze_github_issue_tool`
• Patch generation with `generate_code_patch_tool`
• Implementation guidance with `get_patch_guidance`"""
    
    def _format_health_status(self, health: Any, summary: Dict[str, Any]) -> str:
        """Format health status information."""
        status_icon = "✅" if health.status == "healthy" else "⚠️" if health.status == "degraded" else "❌"
        
        return f"""🏥 **MCP Server Health Status**

{status_icon} **Overall Status**: {health.status.title()}
📊 **Health Trend**: {summary.get('health_trend', 'stable').title()}
✅ **Checks Passed**: {summary.get('checks_passed', 0)} / {summary.get('total_checks', 0)}

⚡ **System Health:**
• CPU: {health.details.get('system', {}).get('cpu_percent', 0):.1f}%
• Memory: {health.details.get('system', {}).get('memory_percent', 0):.1f}%
• Disk Free: {health.details.get('system', {}).get('disk_free_gb', 0):.1f} GB

🌐 **API Status:**
• GitHub: {'✅' if health.checks.get('github_api_ok') else '❌'}
• Google: {'✅' if health.checks.get('google_api_ok') else '❌'}

📊 **Performance:**
• Uptime: {health.details.get('performance', {}).get('uptime_hours', 0):.1f} hours
• Requests: {health.details.get('performance', {}).get('requests_processed', 0):,}
• Avg Response: {health.details.get('performance', {}).get('average_response_time', 0):.3f}s

{'🔧 **Recommendations:**' + chr(10) + chr(10).join([f'• {rec}' for rec in summary.get('recommendations', [])]) if summary.get('recommendations') else '✅ All systems healthy'}"""
    
    def _format_clear_confirmation(self, repo_name: str) -> str:
        """Format clear data confirmation message."""
        return f"""⚠️ **Repository Data Clearing Confirmation Required**

You are about to clear ALL data for repository: **{repo_name}**

This will permanently delete:
• All documentation embeddings
• All issue analysis data  
• All code analysis data
• All PR history data
• ChromaDB collections

🎯 **To proceed, call this tool again with confirm=True:**
`clear_repository_data_tool('{repo_name}', confirm=True)`

⚡ **This action cannot be undone!**"""
    
    def _format_cleanup_result(self, result: Dict[str, Any]) -> str:
        """Format cleanup operation result."""
        if result.get('success'):
            return f"""✅ **Cleanup Completed Successfully**

🗑️ **Data Cleaned:**
• Repository entries: {result.get('repositories_cleaned', 0)}
• Health records: {result.get('health_records_cleaned', 0)}

🕒 **Completed at**: {result.get('cleanup_date', 'Unknown')}

💾 **Resources freed and system optimized**"""
        else:
            return f"❌ **Cleanup Failed**: {result.get('error', 'Unknown error')}"
