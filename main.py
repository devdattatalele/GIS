#!/usr/bin/env python3
"""
GitHub Issue Solver MCP Server - Main Entry Point

A professional Model Context Protocol server for automated GitHub issue resolution,
repository analysis, and intelligent patch generation.

This is the new main entry point that replaces the monolithic server implementation
with a properly structured, maintainable codebase following MCP best practices.

Usage:
    python main.py [--env-file .env]
    
Environment Variables (required):
    GOOGLE_API_KEY - Google API key for embeddings and analysis
    GITHUB_TOKEN - GitHub personal access token
    
Optional Environment Variables:
    GOOGLE_DOCS_ID - Google Docs document ID for analysis reports
    CHROMA_PERSIST_DIR - Custom ChromaDB persistence directory
    LOG_LEVEL - Logging level (DEBUG, INFO, WARNING, ERROR)
    MAX_ISSUES - Maximum issues to process per ingestion (default: 100)
    MAX_PRS - Maximum PRs to process per ingestion (default: 50)
    HEALTH_CHECK_INTERVAL - Health check interval in seconds (default: 300)
    
Features:
    • Multi-step repository ingestion with progress tracking
    • AI-powered issue analysis using RAG (Retrieval-Augmented Generation)
    • Intelligent code patch generation
    • Comprehensive health monitoring and recovery
    • Professional error handling and logging
    • State persistence and recovery
    • Modular, testable architecture
"""

import argparse
import sys
from pathlib import Path
from loguru import logger

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from github_issue_solver import GitHubIssueSolverServer


def setup_logging(log_level: str = "INFO") -> None:
    """Setup Loguru-based logging."""
    logger.remove()  # Remove default handler
    logger.add(
        sys.stderr,
        level=log_level.upper(),
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True,
    )
    logger.info(f"Log level set to {log_level.upper()}")
    

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="GitHub Issue Solver MCP Server",
        epilog="""
Examples:
    python main.py                    # Use default .env file
    python main.py --env-file prod.env  # Use custom environment file
    python main.py --log-level DEBUG   # Enable debug logging
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--env-file",
        type=str,
        help="Path to environment file (default: .env in current directory)"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        default="mcp",
        choices=["mcp", "cli"],
        help="Run mode: 'mcp' for MCP server, 'cli' for interactive terminal (default: mcp)"
    )

    parser.add_argument(
        "--transport",
        type=str,
        default="stdio",
        choices=["stdio", "sse"],
        help="MCP transport protocol (default: stdio)"
    )
    
    parser.add_argument(
        "--investigate",
        type=str,
        metavar="URL",
        help="Launch the 10-phase investigation pipeline directly on a GitHub issue URL (implies --mode cli)"
    )

    parser.add_argument(
        "--version",
        action="version",
        version="GitHub Issue Solver MCP Server 3.0.0"
    )

    return parser.parse_args()


def print_banner() -> None:
    """Print server banner with key information."""
    banner = """
╔══════════════════════════════════════════════════════════════════════════╗
║                    GitHub Issue Solver MCP Server v2.0                  ║
║                                                                          ║
║  🚀 Professional MCP server for automated GitHub issue resolution       ║
║  🤖 AI-powered analysis using RAG with repository knowledge base        ║
║  🔧 Intelligent code patch generation and solution recommendations       ║
║  📊 Comprehensive health monitoring and error recovery                   ║
║  🏗️  Modular architecture following MCP best practices                  ║
║                                                                          ║
║  Ready to revolutionize your GitHub workflow!                           ║
╚══════════════════════════════════════════════════════════════════════════╝
    """
    print(banner, file=sys.stderr)


def print_startup_info() -> None:
    """Print startup information and usage guide."""
    info = """
🎯 Quick Start Guide:

1️⃣ **Initialize Repository Ingestion:**
   start_repository_ingestion('owner/repo')

2️⃣ **Run 4-Step Ingestion Process:**
   • ingest_repository_docs('owner/repo')      - Documentation
   • ingest_repository_code('owner/repo')      - Source code  
   • ingest_repository_issues('owner/repo')    - Issues history
   • ingest_repository_prs('owner/repo')       - PR history

3️⃣ **Analyze Issues & Generate Solutions:**
   • analyze_github_issue_tool('https://github.com/owner/repo/issues/123')
   • generate_code_patch_tool('issue description', 'owner/repo')

4️⃣ **Monitor & Manage:**
   • get_repository_status('owner/repo')       - Check progress
   • get_health_status_tool()                  - Server health
   • list_ingested_repositories()              - View all repos

💡 **Pro Tips:**
   • Each ingestion step can be run independently and retried if needed
   • Health monitoring runs automatically in the background
   • State is persisted across server restarts
   • Use repository validation before ingestion to catch issues early

🔗 **Integration:**
   This server works seamlessly with Claude Desktop, Cursor, and other MCP clients.
   For PR creation, use the official GitHub MCP server in combination with this one.

═══════════════════════════════════════════════════════════════════════════════
    """
    print(info, file=sys.stderr)


def main() -> None:
    """Main entry point for the GitHub Issue Solver."""
    try:
        # Parse command line arguments
        args = parse_arguments()

        # Setup logging
        setup_logging(args.log_level)

        # --- CLI Mode ---
        if args.mode == "cli" or args.investigate:
            from github_issue_solver.cli import CLIApp
            app = CLIApp(env_file=args.env_file, investigate_url=args.investigate)
            app.run()
            return

        # --- MCP Mode (default) ---
        # Print banner
        print_banner()

        # Create and run server
        logger.info("Initializing GitHub Issue Solver MCP Server v2.0")

        # Initialize server with proper error handling
        try:
            server = GitHubIssueSolverServer(env_file=args.env_file)
            logger.info("Server initialized successfully")
        except Exception as e:
            logger.error(f"Server initialization failed: {e}")
            logger.error("Please check your environment variables and configuration")
            sys.exit(1)

        # Print startup information
        print_startup_info()

        # Run server
        logger.info(f"Starting MCP server with {args.transport} transport")
        logger.info("Server is now ready to accept MCP connections")
        logger.info("Use Ctrl+C to gracefully shutdown the server")

        server.run(transport=args.transport)

    except KeyboardInterrupt:
        logger.info("Server shutdown requested by user")
        logger.info("Thank you for using GitHub Issue Solver!")

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error("Please check the logs above for more details")
        sys.exit(1)


if __name__ == "__main__":
    main()
