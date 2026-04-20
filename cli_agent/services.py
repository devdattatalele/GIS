"""Service initialization bridge.

Mirrors the service init from src/github_issue_solver/cli/app.py
but designed for standalone CLI use with optional overrides.
"""

import os
from typing import Optional

from cli_agent.config import setup_sys_path


def initialize_services(
    env_file: Optional[str] = None,
    model: Optional[str] = None,
    provider: Optional[str] = None,
) -> tuple:
    """
    Initialize all services needed by the agent.

    Returns:
        (config, services_dict)
    """
    # Ensure src/ and project root are on sys.path
    setup_sys_path()

    # Apply overrides before Config reads env
    if provider:
        os.environ["LLM_PROVIDER"] = provider
    if model:
        os.environ["LLM_MODEL_NAME"] = model

    # Lazy imports to avoid triggering heavy __init__.py chains
    from github_issue_solver.config import Config
    from github_issue_solver.services.state_manager import StateManager
    from github_issue_solver.services.repository_service import RepositoryService
    from github_issue_solver.services.embedding_service import EmbeddingService
    from github_issue_solver.services.ingestion_service import IngestionService
    from github_issue_solver.services.analysis_service import AnalysisService
    from github_issue_solver.services.patch_service import PatchService
    from github_issue_solver.services.llm_service import LLMService
    from github_issue_solver.services.learning_service import LearningService

    config = Config(env_file)

    services = {}
    services["state_manager"] = StateManager(config)
    services["repository"] = RepositoryService(config)
    services["embedding"] = EmbeddingService(config)
    services["llm"] = LLMService(config)
    services["ingestion"] = IngestionService(
        config,
        services["repository"],
        services["state_manager"],
        services["embedding"],
    )
    services["analysis"] = AnalysisService(
        config,
        services["repository"],
        services["state_manager"],
        services["llm"],
    )
    services["patch"] = PatchService(
        config,
        services["state_manager"],
        services["llm"],
    )
    services["learning"] = LearningService(
        config,
        services["embedding"],
    )

    return config, services
