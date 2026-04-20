# patch_generator.py

import os
import re
import json
import tempfile
import subprocess
import logging
from datetime import datetime
from typing import Dict, List, Any
from dotenv import load_dotenv

# Configure logging
logger = logging.getLogger(__name__)

from github import Github
from langchain_chroma import Chroma
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# --- Configuration ---
load_dotenv()

# Load API keys from .env file
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

# Chroma configuration
# Determine project root and Chroma DB directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", os.path.abspath(os.path.join(PROJECT_ROOT, "chroma_db")))

# Validate required environment variables
required_vars = {
    'GOOGLE_API_KEY': GOOGLE_API_KEY,
    'GITHUB_TOKEN': GITHUB_TOKEN
}

for var_name, var_value in required_vars.items():
    if not var_value:
        raise ValueError(f"Required environment variable {var_name} is not set in .env file")

# --- Helper Functions ---

def _extract_json_from_response(text: str) -> str:
    """
    Aggressively finds and extracts a JSON object from a string.
    Handles markdown code fences, leading/trailing text, and partial objects.
    """
    # Pattern to find JSON enclosed in ```json ... ```
    match = re.search(r'```json\s*\n(.*?)\n\s*```', text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Fallback pattern for a simple JSON object `{...}`
    match = re.search(r'(\{.*\})', text, re.DOTALL)
    if match:
        return match.group(1).strip()
        
    return text # Return original text if no JSON is found

def initialize_chroma_clients(repo_name: str):
    """Initialize Chroma client and vector stores for both collections with repository-specific names."""
    logger.info(f"Initializing Chroma clients for patch generation (repo: {repo_name})...")
    try:
        # Check if Chroma database exists
        if not os.path.exists(CHROMA_PERSIST_DIR):
            raise ValueError(
                f"Chroma database not found at '{CHROMA_PERSIST_DIR}'. "
                f"Please run the ingestion script first."
            )

        # Initialize embeddings - MUST match the embedding provider used during ingestion
        # This is for QUERYING the vector DB only. The LLM (Gemini) is used separately for reasoning.
        embedding_provider = os.getenv("EMBEDDING_PROVIDER", "google").lower()

        if embedding_provider == "fastembed":
            # Use FastEmbed (offline, no API quota needed) - matches ingestion
            from langchain_community.embeddings import FastEmbedEmbeddings
            model_name = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")
            logger.info(f"Using FastEmbed for vector search (offline): {model_name}")
            logger.info("Note: Gemini LLM will still be used for intelligent patch generation")
            embeddings = FastEmbedEmbeddings(model_name=model_name)
        else:
            # Use Google Generative AI embeddings (requires API quota)
            logger.info("Using Google Generative AI embeddings for vector search")
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=GOOGLE_API_KEY
            )

        # Create repository-specific collection names
        # MUST match what ingestion_service.py uses: "prs" and "code"
        safe_repo_name = repo_name.replace('/', '_').replace('-', '_').lower()
        pr_collection_name = f"{safe_repo_name}_prs"    # FIX: Was "_pr_history"
        code_collection_name = f"{safe_repo_name}_code"  # FIX: Was "_repo_code_main"

        logger.info(f"Using collections: {pr_collection_name}, {code_collection_name}")

        # Create vector stores for both collections
        pr_history_store = Chroma(
            embedding_function=embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
            collection_name=pr_collection_name
        )

        repo_code_store = Chroma(
            embedding_function=embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
            collection_name=code_collection_name
        )

        return pr_history_store, repo_code_store
    except Exception as e:
        raise Exception(f"Failed to initialize Chroma clients: {e}")

def query_vector_stores(issue_body: str, pr_history_store, repo_code_store, k: int = 10):
    """Query both vector stores for relevant context with more results for better patch generation."""
    logger.info("Querying vector stores for comprehensive context...")

    try:
        # Query PR history for past solutions
        pr_results = pr_history_store.similarity_search(issue_body, k=k)
        pr_context = []
        for doc in pr_results:
            pr_context.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })

        # Query repository code with MORE results for precise file identification
        code_results = repo_code_store.similarity_search(issue_body, k=k * 2)  # 2x for code
        code_context = []
        for doc in code_results:
            code_context.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })

        logger.info(f"Found {len(pr_context)} relevant PRs and {len(code_context)} code chunks")
        return pr_context, code_context
    except Exception as e:
        raise Exception(f"Failed to query vector stores: {e}")

def format_context_for_llm(pr_context: List[Dict], code_context: List[Dict]) -> str:
    """Format the retrieved context for the LLM prompt with FULL code for precise patches."""
    formatted_context = ""

    # Format PR history context - these show HOW past issues were fixed
    if pr_context:
        formatted_context += "### PAST FIXES (PR History) - Learn from these solutions:\n"
        for i, pr in enumerate(pr_context[:5], 1):  # Top 5 PRs
            pr_number = pr["metadata"].get("pr_number", "unknown")
            pr_title = pr["metadata"].get("pr_title", "")
            # More content for better learning from past fixes
            content = pr['content'][:2000] if len(pr['content']) > 2000 else pr['content']
            formatted_context += f"**PR #{pr_number}: {pr_title}**\n"
            formatted_context += f"{content}\n\n"

    # Format code context with FULL file paths and line numbers for PRECISE patches
    if code_context:
        formatted_context += "### RELEVANT CODE (Use these exact file paths and line numbers):\n"
        for i, code in enumerate(code_context[:10], 1):  # Top 10 code chunks
            file_path = code["metadata"].get("filePath", code["metadata"].get("source", "unknown"))
            function_name = code["metadata"].get("functionName", "")
            start_line = code["metadata"].get("start_line", "?")
            end_line = code["metadata"].get("end_line", "?")

            func_info = f" - Function: `{function_name}`" if function_name else ""
            formatted_context += f"**File: `{file_path}` (lines {start_line}-{end_line}){func_info}:**\n"
            # Include MORE code content for accurate patch generation
            content = code['content'][:3000] if len(code['content']) > 3000 else code['content']
            formatted_context += f"```\n{content}\n```\n\n"

    return formatted_context

def generate_patch_for_issue(issue_body: str, repo_full_name: str = None) -> Dict[str, Any]:
    """
    Generate patch suggestions for a GitHub issue.
    
    Args:
        issue_body: The body text of the GitHub issue
        repo_full_name: Repository name for context (required for repository-specific collections)
        
    Returns:
        Dictionary with filesToUpdate and summaryOfChanges
    """
    logger.info("Starting patch generation process...")
    
    if not repo_full_name:
        raise ValueError("repo_full_name is required for patch generation")
    
    try:
        # Initialize vector stores with repository-specific collections
        pr_history_store, repo_code_store = initialize_chroma_clients(repo_full_name)
        
        # Query both collections
        pr_context, code_context = query_vector_stores(issue_body, pr_history_store, repo_code_store)
        
        # Format context for LLM
        context_text = format_context_for_llm(pr_context, code_context)
        
        # Initialize LLM with rate limit handling
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            temperature=0.1, 
            google_api_key=GOOGLE_API_KEY,
            max_retries=2,  # Reduce retries
            request_timeout=30  # Shorter timeout
        )
        
        # Construct the prompt with Snitch methodology for deep analysis
        prompt = f"""You are **Snitch**, an advanced patch generator. Your patches must be EXACT and IMMEDIATELY APPLICABLE.

## YOUR METHODOLOGY:

**Step 1: INTERPRET the Issue**
- What is the user reporting?
- What behavior needs to change?
- What are 2-3 ways to fix this?

**Step 2: CLARITY WINDOW**
- Lock in: The fix needs to change [X] to [Y] in [file]
- Keep this focus throughout

**Step 3: PARALLEL APPROACHES**
- Approach A: Minimal fix (change just the buggy line)
- Approach B: Robust fix (add validation/error handling)
- Approach C: Comprehensive fix (refactor if needed)

**Step 4: SELECT & EXECUTE**
- Choose the best approach based on code context
- Generate EXACT patch using REAL code from context

---

## GITHUB ISSUE TO FIX:
{issue_body}

## RELEVANT CODEBASE CONTEXT (Use EXACT paths and code from here):
{context_text}

---

## CRITICAL RULES:
1. Use ONLY file paths from "RELEVANT CODE" section above
2. Use EXACT variable names from the code (no inventing prefixes like `schema.`)
3. Your patch must match the actual code shown
4. Generate unified diff format for `git apply`

---

## OUTPUT FORMAT (JSON):

```json
{{
  "interpretations": [
    "The issue could mean...",
    "Alternatively, it might be..."
  ],
  "clarity_window": "The core fix is to change X to Y in file Z",
  "approaches": {{
    "minimal": "Just fix line X",
    "robust": "Add validation + fix",
    "comprehensive": "Refactor the function"
  }},
  "selected_approach": "minimal/robust/comprehensive",
  "filesToUpdate": [
    {{
      "filePath": "exact/path/from/context.ext",
      "startLine": 21,
      "endLine": 30,
      "functionName": "functionName or null",
      "operation": "modify",
      "currentCode": "EXACT code currently in file (from context)",
      "patch": "@@ -21,10 +21,12 @@\\n context\\n-old line\\n+new line\\n context",
      "explanation": "Why this change fixes the issue"
    }}
  ],
  "summaryOfChanges": "Clear summary of what was changed and why",
  "confidence": "high/medium/low",
  "changelog": "1) Interpreted as... 2) Found code in... 3) Applied fix..."
}}
```

**Patch Format:**
- @@ -START,COUNT +START,COUNT @@ header REQUIRED
- `-` = removed lines, `+` = added lines, ` ` = context
- Include 3+ lines of context before/after

**If insufficient context:**
```json
{{"filesToUpdate": [], "summaryOfChanges": "Could not generate patch: [reason]", "confidence": "low"}}
```

Generate the patches now:"""

        # Get LLM response with error handling
        try:
            response = llm.invoke(prompt)
        except Exception as e:
            error_message = str(e)
            if "429" in error_message or "quota" in error_message.lower():
                logger.warning("⚠️ Google API rate limit exceeded for patch generation")
                return {
                    "filesToUpdate": [],
                    "summaryOfChanges": "Patch generation skipped due to API rate limits. Please try again later or resolve manually."
                }
            raise e
        
        # Parse JSON response
        try:
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Use the new helper to clean the response before parsing
            json_str = _extract_json_from_response(response_text)
            
            patch_data = json.loads(json_str)
            
            # Validate the structure
            if "filesToUpdate" not in patch_data or "summaryOfChanges" not in patch_data:
                raise ValueError("Invalid patch data structure: Missing required keys 'filesToUpdate' or 'summaryOfChanges'.")
            
            logger.info(f"Successfully generated and parsed patch for {len(patch_data['filesToUpdate'])} files.")
            return patch_data
            
        except json.JSONDecodeError as e:
            error_message = f"Error: Failed to parse a valid JSON patch from the AI's response. Please try again. Error details: {e}"
            logger.error(error_message)
            logger.error(f"Raw response was: {response_text}")
            return {
                "filesToUpdate": [],
                "summaryOfChanges": error_message
            }
    
    except Exception as e:
        logger.error(f"Error in patch generation: {e}")
        return {
            "filesToUpdate": [],
            "summaryOfChanges": f"Error during patch generation: {str(e)}"
        }

# The main execution block is removed to convert this file into a library module. 