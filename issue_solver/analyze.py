# analyze_issue.py
import os
import re
import json
import argparse
import logging
from datetime import datetime
from dotenv import load_dotenv

# Configure logging
logger = logging.getLogger(__name__)

from github import Github
# --- LangChain Imports ---
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools.retriever import create_retriever_tool
from langchain_core.prompts import ChatPromptTemplate

# --- Google Docs API Imports ---
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# --- Patch Generator Import ---
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# --- Configuration ---
# Load environment variables from .env file if it exists
# load_dotenv() will not override existing environment variables set via Docker -e flags
load_dotenv()

# Load API keys from environment (either from .env file or Docker -e flags)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GOOGLE_DOCS_ID = os.getenv("GOOGLE_DOCS_ID")

# Chroma configuration
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", os.path.abspath(os.path.join(PROJECT_ROOT, "chroma_db")))

# All collection names - MUST match what ingestion_service.py uses!
# These are the ACTUAL names used during ingestion (from ingestion_service.py lines 120, 138, 159, 180)
COLLECTION_DOCS = "documentation"  # Matches ingestion
COLLECTION_CODE = "code"           # FIX: Was "repo_code_main", ingestion uses "code"
COLLECTION_ISSUES = "issues"       # FIX: Was "issues_history", ingestion uses "issues"
COLLECTION_PRS = "prs"             # FIX: Was "pr_history", ingestion uses "prs"

# Patch generator configuration
ENABLE_PATCH_GENERATION = os.getenv("ENABLE_PATCH_GENERATION", "true").lower() == "true"
MAX_COMPLEXITY_FOR_AUTO_PR = int(os.getenv("MAX_COMPLEXITY_FOR_AUTO_PR", "4"))

# Validate required environment variables (only check truly required ones)
required_vars = {
    'GOOGLE_API_KEY': GOOGLE_API_KEY,
    'GITHUB_TOKEN': GITHUB_TOKEN
}

for var_name, var_value in required_vars.items():
    if not var_value:
        raise ValueError(f"Required environment variable {var_name} is not set in environment")

# GOOGLE_DOCS_ID is optional - log warning if not present
if not GOOGLE_DOCS_ID:
    logger.warning("GOOGLE_DOCS_ID not set - analysis results will not be saved to Google Docs")

# Google Docs API scopes
SCOPES = ["https://www.googleapis.com/auth/documents"]

# --- Helper Functions ---

def parse_github_url(url: str):
    """Parses a GitHub issue URL to get owner, repo, and issue number."""
    match = re.search(r"github\.com/([^/]+)/([^/]+)/issues/(\d+)", url)
    if not match:
        raise ValueError("Invalid GitHub issue URL format. Expected: https://github.com/owner/repo/issues/number")
    return match.group(1), match.group(2), int(match.group(3))

def get_github_issue(owner: str, repo_name: str, issue_number: int):
    """Fetches issue data from GitHub."""
    logger.info(f"Fetching issue '{owner}/{repo_name}#{issue_number}' from GitHub...")
    try:
        g = Github(GITHUB_TOKEN)
        repo = g.get_repo(f"{owner}/{repo_name}")
        issue = repo.get_issue(number=issue_number)
        return issue
    except Exception as e:
        raise Exception(f"Failed to fetch GitHub issue: {e}")

def initialize_chroma_retriever(repo_name: str = None):
    """Initializes the Chroma vector store and retriever tool with repository-specific collection."""
    logger.info("Initializing Chroma vector store and retriever...")
    try:
        # Check if Chroma database exists
        if not os.path.exists(CHROMA_PERSIST_DIR):
            raise ValueError(
                f"Chroma database not found at '{CHROMA_PERSIST_DIR}'. "
                f"Please run the ingestion script first."
            )

        # Initialize embeddings - MUST match the provider used during ingestion
        embedding_provider = os.getenv("EMBEDDING_PROVIDER", "google").lower()

        if embedding_provider == "google":
            logger.info("Using Google embeddings for retriever")
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=GOOGLE_API_KEY
            )
        else:
            from langchain_community.embeddings import FastEmbedEmbeddings
            model_name = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")
            logger.info(f"Using FastEmbed for retriever: {model_name}")
            embeddings = FastEmbedEmbeddings(model_name=model_name)
        
        # Create repository-specific collection name for issues
        if repo_name:
            safe_repo_name = repo_name.replace('/', '_').replace('-', '_').lower()
            collection_name = f"{safe_repo_name}_{COLLECTION_ISSUES}"
        else:
            collection_name = COLLECTION_ISSUES
            
        logger.info(f"Loading Chroma collection: {collection_name}")
        chroma_store = Chroma(
            embedding_function=embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
            collection_name=collection_name
        )
        
        # Create retriever
        retriever = chroma_store.as_retriever(
            search_kwargs={"k": 5}
        )
        
        # This creates the tool the agent can use
        tool = create_retriever_tool(
            retriever,
            "github_knowledge_base_search",
            "Search the knowledge base for relevant past issues and documentation. Use this to find context for a new GitHub issue.",
        )
        return tool
    except Exception as e:
        raise Exception(f"Failed to initialize Chroma retriever: {e}")

def create_langchain_agent(issue):
    """
    Analyzes a GitHub issue using comprehensive RAG across ALL ingested collections.
    Searches code, docs, issues, and PRs to provide PRECISE file locations and fixes.
    """
    logger.info("Initializing Comprehensive Multi-Collection RAG Analysis...")
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.1,  # Lower temperature for more precise outputs
            google_api_key=GOOGLE_API_KEY,
            max_retries=2,
            request_timeout=60
        )

        repo_name = issue.repository.full_name
        safe_repo_name = repo_name.replace('/', '_').replace('-', '_').lower()

        # Initialize embeddings - MUST match the provider used during ingestion
        embedding_provider = os.getenv("EMBEDDING_PROVIDER", "google").lower()

        if embedding_provider == "google":
            # Use Google Generative AI embeddings
            logger.info("Using Google embeddings for vector search")
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=GOOGLE_API_KEY,
            )
        else:
            # Use FastEmbed (offline, no API quota needed)
            from langchain_community.embeddings import FastEmbedEmbeddings
            model_name = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")
            logger.info(f"Using FastEmbed for vector search (offline): {model_name}")
            embeddings = FastEmbedEmbeddings(model_name=model_name)

        # Build query from issue content
        issue_title = issue.title
        issue_body = issue.body or ""
        base_query = f"{issue_title}\n{issue_body}"

        # Extract key technical terms for better code search
        technical_keywords = _extract_technical_keywords(issue_title, issue_body)
        code_query = f"{issue_title} {' '.join(technical_keywords)}"

        logger.info(f"Searching with keywords: {technical_keywords[:10]}")

        # === SEARCH ALL 4 COLLECTIONS ===
        all_context = {}
        retrieved_files = []  # Track all retrieved file paths for verification

        # 1. Search CODE collection (MOST IMPORTANT for precise fixes)
        try:
            code_collection = f"{safe_repo_name}_{COLLECTION_CODE}"
            code_store = Chroma(
                embedding_function=embeddings,
                persist_directory=CHROMA_PERSIST_DIR,
                collection_name=code_collection,
            )
            code_retriever = code_store.as_retriever(search_kwargs={"k": 10})
            code_docs = code_retriever.get_relevant_documents(code_query)

            # Format code results with file paths and line numbers
            code_context = []
            for doc in code_docs:
                meta = doc.metadata
                file_path = meta.get('filePath', meta.get('source', 'unknown'))
                func_name = meta.get('functionName', '')
                start_line = meta.get('start_line', '?')
                end_line = meta.get('end_line', '?')

                # Track retrieved files for hallucination detection
                retrieved_files.append(file_path)

                code_entry = f"**File: {file_path}** (lines {start_line}-{end_line})"
                if func_name:
                    code_entry += f" - Function: {func_name}"
                code_entry += f"\n```\n{doc.page_content[:2000]}\n```"
                code_context.append(code_entry)

            all_context['code'] = "\n\n".join(code_context) if code_context else "No relevant code found."
            logger.info(f"Found {len(code_docs)} relevant code chunks")
            logger.info(f"Retrieved files for grounding: {retrieved_files[:5]}...")
        except Exception as e:
            logger.warning(f"Code collection search failed: {e}")
            all_context['code'] = "Code collection not available."

        # 2. Search DOCS collection
        try:
            docs_collection = f"{safe_repo_name}_{COLLECTION_DOCS}"
            docs_store = Chroma(
                embedding_function=embeddings,
                persist_directory=CHROMA_PERSIST_DIR,
                collection_name=docs_collection,
            )
            docs_retriever = docs_store.as_retriever(search_kwargs={"k": 5})
            doc_results = docs_retriever.get_relevant_documents(base_query)

            docs_context = []
            for doc in doc_results:
                source = doc.metadata.get('source', 'unknown')
                docs_context.append(f"**Doc: {source}**\n{doc.page_content[:1500]}")

            all_context['docs'] = "\n\n".join(docs_context) if docs_context else "No relevant documentation found."
            logger.info(f"Found {len(doc_results)} relevant docs")
        except Exception as e:
            logger.warning(f"Docs collection search failed: {e}")
            all_context['docs'] = "Documentation collection not available."

        # 3. Search ISSUES collection (for similar past issues)
        try:
            issues_collection = f"{safe_repo_name}_{COLLECTION_ISSUES}"
            issues_store = Chroma(
                embedding_function=embeddings,
                persist_directory=CHROMA_PERSIST_DIR,
                collection_name=issues_collection,
            )
            issues_retriever = issues_store.as_retriever(search_kwargs={"k": 5})
            issue_results = issues_retriever.get_relevant_documents(base_query)

            issues_context = []
            for doc in issue_results:
                source = doc.metadata.get('source', 'unknown')
                issue_num = doc.metadata.get('issue_number', '')
                issue_title_found = doc.metadata.get('issue_title', '')
                issues_context.append(f"**{source}**: {issue_title_found}\n{doc.page_content[:1000]}")

            all_context['issues'] = "\n\n".join(issues_context) if issues_context else "No similar issues found."
            logger.info(f"Found {len(issue_results)} similar issues")
        except Exception as e:
            logger.warning(f"Issues collection search failed: {e}")
            all_context['issues'] = "Issues collection not available."

        # 4. Search PR HISTORY collection (for past fixes)
        try:
            prs_collection = f"{safe_repo_name}_{COLLECTION_PRS}"
            prs_store = Chroma(
                embedding_function=embeddings,
                persist_directory=CHROMA_PERSIST_DIR,
                collection_name=prs_collection,
            )
            prs_retriever = prs_store.as_retriever(search_kwargs={"k": 5})
            pr_results = prs_retriever.get_relevant_documents(base_query)

            prs_context = []
            for doc in pr_results:
                pr_num = doc.metadata.get('pr_number', '?')
                pr_title = doc.metadata.get('pr_title', '')
                prs_context.append(f"**PR #{pr_num}**: {pr_title}\n{doc.page_content[:1500]}")

            all_context['prs'] = "\n\n".join(prs_context) if prs_context else "No relevant PRs found."
            logger.info(f"Found {len(pr_results)} relevant PRs")
        except Exception as e:
            logger.warning(f"PRs collection search failed: {e}")
            all_context['prs'] = "PR history collection not available."

        # === BUILD COMPREHENSIVE PROMPT WITH SNITCH METHODOLOGY ===
        template = """You are **Snitch**, an advanced intelligence built to solve GitHub issues for the '{repo_full_name}' repository.

## YOUR METHODOLOGY (Follow this thinking process):

**Step 1: INTERPRET** - Draw 3-4 interpretations of what the issue is actually about:
- What is the user experiencing?
- What could be causing this?
- What are different ways to interpret this bug?

**Step 2: CLARITY WINDOW** - Lock in your understanding:
- What is the CORE problem?
- What is the expected vs actual behavior?
- Keep this clarity throughout your analysis.

**Step 3: PARALLEL ANALYSIS** - Consider 3 different approaches:
- Approach A: [Most likely cause based on code]
- Approach B: [Alternative interpretation]
- Approach C: [Edge case or deeper issue]

**Step 4: INVESTIGATE** - Search the SOURCE CODE below for evidence:
- Find the relevant files and functions
- Quote EXACT code as evidence
- Trace the data flow if needed

**Step 5: ROOT CAUSE** - Based on evidence, determine:
- WHY does this bug occur?
- WHAT specific line/condition causes it?
- HOW does data flow to this point?

**Step 6: SOLUTION** - Propose fix with confidence level:
- EXACT file path and line numbers
- EXACT code change (before → after)
- Verify variable names match the code

---

## CRITICAL: GROUNDED ANALYSIS ONLY
You MUST base your analysis ONLY on the code shown below. Do NOT hallucinate.
- If a variable is named `dateFormat`, use `dateFormat` - NOT `schema.dateFormat`
- If you cannot find relevant code, say "Insufficient context" - do NOT make up paths

---

## SOURCE CODE FROM REPOSITORY:
{code_context}

## DOCUMENTATION:
{docs_context}

## SIMILAR PAST ISSUES:
{issues_context}

## RELATED PULL REQUESTS (Past fixes):
{prs_context}

---

## CURRENT ISSUE TO ANALYZE:
**Title:** {issue_title}
**Body:** {issue_body}
**URL:** {issue_url}

---

## YOUR ANALYSIS (Show your thinking):

### 1. INTERPRETATIONS (What could this issue mean?)
Think through 3-4 possible interpretations...

### 2. CLARITY WINDOW (Lock in understanding)
The core problem is...

### 3. PARALLEL CASES (3 approaches considered)
- Case A: ...
- Case B: ...
- Case C: ...

### 4. EVIDENCE (Quote from SOURCE CODE above)
Found in file X, lines Y-Z: ```actual code```

### 5. ROOT CAUSE
The bug occurs because...

### 6. SOLUTION
Change file X, line Y from... to...

---

## FINAL OUTPUT (JSON):

```json
{{
  "summary": "One sentence describing the problem",
  "interpretations": [
    "Interpretation 1: ...",
    "Interpretation 2: ...",
    "Interpretation 3: ..."
  ],
  "clarity_window": "The core problem is X causing Y when Z happens",
  "parallel_cases": {{
    "case_a": {{"hypothesis": "...", "likelihood": "high/medium/low"}},
    "case_b": {{"hypothesis": "...", "likelihood": "high/medium/low"}},
    "case_c": {{"hypothesis": "...", "likelihood": "high/medium/low"}}
  }},
  "selected_case": "case_a",
  "code_found": true,
  "quoted_evidence": "EXACT code from SOURCE CODE section",
  "root_cause": "Technical explanation with specific line references",
  "affected_files": [
    {{
      "file_path": "EXACT path from SOURCE CODE",
      "line_numbers": "XX-YY",
      "current_code": "exact current code",
      "suggested_fix": "exact fix with same variable names",
      "explanation": "why this fixes it"
    }}
  ],
  "proposed_solution": "Step-by-step fix with code blocks",
  "complexity": 1-5,
  "similar_issues": ["issue #X", "PR #Y"],
  "confidence": "high/medium/low",
  "changelog": "Analysis steps taken: 1) Interpreted issue as... 2) Found code in... 3) Identified root cause..."
}}
```

**VERIFICATION BEFORE RESPONDING:**
- [ ] Every file_path appears in SOURCE CODE section above
- [ ] Every variable name matches EXACTLY (no invented prefixes)
- [ ] I quoted actual code, not guessed code
- [ ] My changelog shows my reasoning process

**Final Answer (JSON):**"""

        prompt = ChatPromptTemplate.from_template(template)

        from langchain_core.output_parsers import StrOutputParser
        rag_chain = prompt | llm | StrOutputParser()

        logger.info("Running comprehensive RAG analysis...")

        inputs = {
            "code_context": all_context['code'],
            "docs_context": all_context['docs'],
            "issues_context": all_context['issues'],
            "prs_context": all_context['prs'],
            "issue_title": issue_title,
            "issue_body": issue_body or "No body provided.",
            "issue_url": issue.html_url,
            "repo_full_name": repo_name
        }

        response = rag_chain.invoke(inputs)

        # === GROUNDING VERIFICATION ===
        # Check if the LLM's response references files that were actually retrieved
        try:
            # Try to parse response as JSON for verification
            import re
            json_match = re.search(r'```json\s*\n([\s\S]*?)\n\s*```', response)
            if json_match:
                json_str = json_match.group(1).strip()
            else:
                json_match = re.search(r'(\{[\s\S]*\})', response)
                json_str = json_match.group(1) if json_match else None

            if json_str:
                analysis_dict = json.loads(json_str)
                verified_analysis = _verify_grounding(analysis_dict, retrieved_files)

                # Log any grounding warnings
                if '_grounding_warnings' in verified_analysis:
                    for warning in verified_analysis['_grounding_warnings']:
                        logger.warning(f"🚨 {warning}")

                # Return the verified analysis as JSON string
                return json.dumps(verified_analysis)
        except Exception as verify_error:
            logger.warning(f"Could not verify grounding (will return raw response): {verify_error}")

        return response

    except Exception as e:
        error_message = str(e)

        if "429" in error_message or "quota" in error_message.lower():
            logger.warning("⚠️ Google API rate limit exceeded. Using fallback analysis...")
            return create_fallback_analysis(issue)

        logger.error(f"RAG chain error: {error_message}")
        raise Exception(f"Failed to run LangChain RAG chain: {e}")


def _verify_grounding(analysis: dict, retrieved_files: list) -> dict:
    """
    Verify that the LLM's analysis is grounded in actual retrieved code.
    Now KEEPS all files but adds warnings for unverified ones.
    Also adds retrieved files if LLM missed them.
    """
    verified_files = []
    warnings = []
    seen_files = set()

    # Process LLM's suggested files - KEEP all but mark confidence
    for file_info in analysis.get('affected_files', []):
        file_path = file_info.get('file_path', '')
        if not file_path:
            continue

        seen_files.add(file_path.lower())

        # Check if this file was actually in the retrieved context
        found = any(file_path in rf for rf in retrieved_files)

        if found:
            file_info['_verified'] = True
            verified_files.append(file_info)
        else:
            # Check for partial matches (in case of path differences)
            basename = file_path.split('/')[-1] if '/' in file_path else file_path
            partial_match = any(basename in rf for rf in retrieved_files)

            if partial_match:
                file_info['_warning'] = f"Path inferred - verify exact location"
                file_info['_verified'] = True
                verified_files.append(file_info)
            else:
                # KEEP the file but mark as unverified (might be correct but not in RAG)
                file_info['_verified'] = False
                file_info['_warning'] = "Not in knowledge base - verify manually"
                verified_files.append(file_info)
                warnings.append(f"Unverified: '{file_path}' - not in retrieved context")

    # ADD retrieved files that LLM missed (these are definitely relevant)
    for rf in retrieved_files[:5]:  # Top 5 most relevant
        rf_lower = rf.lower()
        if not any(rf_lower in seen.lower() for seen in seen_files):
            verified_files.append({
                "file_path": rf,
                "line_numbers": "check full file",
                "description": "Retrieved by RAG as semantically relevant",
                "_verified": True,
                "_source": "rag_retrieval"
            })

    if warnings:
        analysis['_grounding_warnings'] = warnings
        if len(warnings) > len(verified_files) // 2:
            analysis['confidence'] = 'low'
        analysis['_note'] = "Some files need manual verification. RAG-retrieved files have been added."

    analysis['affected_files'] = verified_files
    return analysis


def _extract_technical_keywords(title: str, body: str) -> list:
    """Extract technical keywords from issue for better code search."""
    import re

    combined = f"{title} {body}".lower()

    # Common programming patterns to look for
    patterns = [
        r'\b([A-Z][a-z]+[A-Z][a-zA-Z]*)\b',  # CamelCase (component names)
        r'\b([a-z]+_[a-z_]+)\b',              # snake_case (function names)
        r'\b([a-z]+\.[a-z]+)\b',              # file.ext patterns
        r'`([^`]+)`',                          # Code in backticks
        r'\b(error|bug|crash|fail|issue|problem)\b',
        r'\b(function|method|class|component|module)\b',
        r'\b([a-zA-Z]+Error|[a-zA-Z]+Exception)\b',  # Error types
    ]

    keywords = set()
    for pattern in patterns:
        matches = re.findall(pattern, combined, re.IGNORECASE)
        keywords.update(matches)

    # Also extract words from title (usually most relevant)
    title_words = [w for w in title.split() if len(w) > 3 and w.isalnum()]
    keywords.update(title_words)

    # Filter out common non-technical words
    stop_words = {'the', 'and', 'for', 'with', 'this', 'that', 'from', 'have', 'been', 'when', 'where', 'what', 'does', 'should', 'would', 'could'}
    keywords = [k for k in keywords if k.lower() not in stop_words]

    return list(keywords)[:20]  # Limit to top 20 keywords

def create_fallback_analysis(issue):
    """
    Create analysis by searching knowledge base directly when LLM rate limits hit.
    Still searches all collections to provide useful context.
    """
    logger.info("Creating fallback analysis with direct knowledge base search...")

    title = issue.title
    body = issue.body or ""
    repo_name = issue.repository.full_name
    safe_repo_name = repo_name.replace('/', '_').replace('-', '_').lower()

    # Extract keywords for search
    keywords = _extract_technical_keywords(title, body)
    search_query = f"{title} {' '.join(keywords[:5])}"

    # Try to search code collection directly
    affected_files = []
    code_snippets = []

    try:
        # Initialize embeddings - MUST match the provider used during ingestion
        embedding_provider = os.getenv("EMBEDDING_PROVIDER", "google").lower()

        if embedding_provider == "google":
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            logger.info("Fallback using Google embeddings for vector search")
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=GOOGLE_API_KEY,
            )
        else:
            from langchain_community.embeddings import FastEmbedEmbeddings
            model_name = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")
            logger.info(f"Fallback using FastEmbed for vector search: {model_name}")
            embeddings = FastEmbedEmbeddings(model_name=model_name)

        # Search CODE collection
        try:
            code_collection = f"{safe_repo_name}_{COLLECTION_CODE}"
            code_store = Chroma(
                embedding_function=embeddings,
                persist_directory=CHROMA_PERSIST_DIR,
                collection_name=code_collection,
            )
            code_retriever = code_store.as_retriever(search_kwargs={"k": 8})
            code_docs = code_retriever.get_relevant_documents(search_query)

            for doc in code_docs[:5]:
                meta = doc.metadata
                file_path = meta.get('filePath', meta.get('source', 'unknown'))
                start_line = meta.get('start_line', '?')
                end_line = meta.get('end_line', '?')
                func_name = meta.get('functionName', '')

                affected_files.append({
                    "file_path": file_path,
                    "line_numbers": f"{start_line}-{end_line}",
                    "description": f"Contains relevant code{f' - function: {func_name}' if func_name else ''}"
                })
                code_snippets.append(f"**{file_path}:{start_line}**\n```\n{doc.page_content[:500]}...\n```")

            logger.info(f"Fallback found {len(code_docs)} code matches")
        except Exception as e:
            logger.warning(f"Fallback code search failed: {e}")

        # Search ISSUES collection for similar issues
        similar_issues = []
        try:
            issues_collection = f"{safe_repo_name}_{COLLECTION_ISSUES}"
            issues_store = Chroma(
                embedding_function=embeddings,
                persist_directory=CHROMA_PERSIST_DIR,
                collection_name=issues_collection,
            )
            issues_retriever = issues_store.as_retriever(search_kwargs={"k": 3})
            issue_docs = issues_retriever.get_relevant_documents(search_query)

            for doc in issue_docs:
                issue_num = doc.metadata.get('issue_number', '')
                if issue_num:
                    similar_issues.append(f"issue #{issue_num}")
        except Exception as e:
            logger.warning(f"Fallback issues search failed: {e}")

    except Exception as e:
        logger.warning(f"Fallback knowledge base search failed: {e}")

    # Build the response
    summary = f"{title}"

    if affected_files:
        proposed_solution = f"""## Potentially Relevant Files Found

Based on knowledge base search, these files may be related to the issue:

{chr(10).join(code_snippets[:3])}

## Recommended Next Steps

1. **Review the files listed above** - They were found by semantic search matching the issue description
2. **Search for specific keywords**: `{', '.join(keywords[:5])}`
3. **Trace the code path** from the entry point to the error location
4. **Check related tests** for expected behavior

*Note: This is a fallback analysis due to API rate limits. For full LLM-powered analysis, please retry later.*"""
    else:
        proposed_solution = f"""## Analysis Required

The issue mentions: {', '.join(keywords[:8])}

**Recommended search patterns:**
- Search codebase for: `{keywords[0] if keywords else title.split()[0]}`
- Look for files related to: {title}

*Note: Knowledge base search returned limited results. The repository may need re-ingestion or the issue may involve new/undocumented functionality.*

*This is a fallback analysis due to API rate limits. For full analysis, please retry later.*"""

    # Estimate complexity based on issue content
    complexity = 3
    if len(body) > 1000 or len(affected_files) > 3:
        complexity = 4
    if any(word in body.lower() for word in ['architecture', 'refactor', 'breaking', 'migration']):
        complexity = 5
    if any(word in body.lower() for word in ['typo', 'documentation', 'readme']):
        complexity = 1

    # Build Snitch-style response for consistency
    fallback_json = {
        "summary": summary,
        "interpretations": [
            f"Issue might be related to: {keywords[0] if keywords else 'unknown'}",
            f"Could involve files matching: {', '.join(keywords[:3])}",
            "May require deeper investigation of data flow"
        ],
        "clarity_window": f"Core problem appears to be: {title}",
        "parallel_cases": {
            "case_a": {"hypothesis": "Direct bug in identified files", "likelihood": "medium"},
            "case_b": {"hypothesis": "Configuration or integration issue", "likelihood": "low"},
            "case_c": {"hypothesis": "Edge case not handled", "likelihood": "low"}
        },
        "selected_case": "case_a",
        "code_found": len(affected_files) > 0,
        "quoted_evidence": code_snippets[0] if code_snippets else "No code found in knowledge base",
        "root_cause": "Requires manual investigation (fallback mode due to rate limits)",
        "affected_files": affected_files,
        "proposed_solution": proposed_solution,
        "complexity": complexity,
        "similar_issues": similar_issues,
        "confidence": "low" if not affected_files else "medium",
        "changelog": f"Fallback analysis: 1) Extracted keywords: {keywords[:5]}, 2) Searched code collection, 3) Found {len(affected_files)} potential files"
    }

    return json.dumps(fallback_json)

def parse_agent_output(raw_output: str):
    """Extracts and parses the JSON from the agent's raw output string."""
    logger.info("Parsing agent's JSON output...")
    logger.info(f"Raw output length: {len(raw_output)}")
    logger.info(f"Raw output preview: {raw_output[:200]}...")
    
    try:
        # Handle empty or invalid output
        if not raw_output or raw_output.strip() == "":
            logger.warning("Empty output received from agent")
            return {
                "summary": "Agent returned empty response",
                "proposed_solution": "Please re-run the analysis or check the issue details",
                "complexity": 3,
                "similar_issues": []
            }
        
        # Handle agent timeout message
        if "Agent stopped due to iteration limit or time limit" in raw_output:
            logger.warning("Agent hit iteration limit")
            return {
                "summary": "Analysis timed out due to complexity",
                "proposed_solution": "The issue requires manual analysis as the automated agent exceeded time limits",
                "complexity": 5,
                "similar_issues": []
            }
        
        # Try to find JSON block within ```json ... ```
        json_match = re.search(r"```json\s*\n([\s\S]*?)\n\s*```", raw_output)
        if json_match:
            json_str = json_match.group(1).strip()
            logger.info(f"Found JSON block: {json_str[:100]}...")
            return json.loads(json_str)
        
        # Try to find JSON block within ``` ... ```
        json_match = re.search(r"```\s*\n([\s\S]*?)\n\s*```", raw_output)
        if json_match:
            json_str = json_match.group(1).strip()
            logger.info(f"Found code block: {json_str[:100]}...")
            return json.loads(json_str)
        
        # Try to find JSON object directly (look for { ... })
        json_match = re.search(r"(\{[\s\S]*\})", raw_output)
        if json_match:
            json_str = json_match.group(1)
            logger.info(f"Found JSON object: {json_str[:100]}...")
            return json.loads(json_str)
        
        # Extract from "Final Answer:" section
        final_answer_match = re.search(r"Final Answer:\s*([\s\S]*?)(?:\n\n|$)", raw_output)
        if final_answer_match:
            answer_text = final_answer_match.group(1).strip()
            logger.info(f"Found Final Answer: {answer_text[:100]}...")
            # Try to parse as JSON
            return json.loads(answer_text)
        
        # If no JSON found, create a summary from the text
        logger.info("No JSON found, creating summary from text")
        return {
            "summary": "Could not parse structured response from agent",
            "proposed_solution": f"Raw agent output: {raw_output[:500]}...",
            "complexity": 3,
            "similar_issues": []
        }
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        logger.error(f"Attempted to parse: {json_str[:200] if 'json_str' in locals() else 'No JSON string found'}...")
        
        # Return a default structure with some extracted info
        return {
            "summary": "Could not parse agent response as JSON",
            "proposed_solution": f"Agent provided response but JSON parsing failed. Raw output: {raw_output[:300]}...",
            "complexity": 3,
            "similar_issues": []
        }


def generate_patches_for_issue(issue, analysis):
    """Generate patches for the issue if conditions are met."""
    logger.info("Evaluating issue for patch generation...")
    
    if not ENABLE_PATCH_GENERATION:
        logger.info("Patch generation is disabled (ENABLE_PATCH_GENERATION=false)")
        return None
    
    try:
        complexity = analysis.get('complexity', 5)
        issue_body = f"Title: {issue.title}\n\nBody: {issue.body or 'No description provided.'}"
        
        # Import patch generation function
        from .patch import generate_patch_for_issue
        
        # Generate patches only (PR creation is now handled by the official GitHub server)
        result = generate_patch_for_issue(
            issue_body=issue_body,
            repo_full_name=issue.repository.full_name
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error in patch generation: {e}")
        return {
            "patch_data": {"filesToUpdate": [], "summaryOfChanges": f"Error: {str(e)}"},
            "pr_url": f"Patch generation failed: {str(e)}",
            "created_pr": False
        }

def append_to_google_doc(text_to_append: str):
    """Handles authentication and appends text to the specified Google Doc."""
    try:
        creds = None
        if os.path.exists("token.json"):
            creds = Credentials.from_authorized_user_file("token.json", SCOPES)
        
        # If there are no (valid) credentials, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not os.path.exists("credentials.json"):
                    raise FileNotFoundError("credentials.json file not found. Please download it from Google Cloud Console.")
                flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
                creds = flow.run_local_server(port=0)
            # Save the credentials for the next run
            with open("token.json", "w") as token:
                token.write(creds.to_json())
        
        logger.info("Appending analysis to Google Doc...")
        service = build("docs", "v1", credentials=creds)
        requests = [
            {
                "insertText": {
                    "location": {"index": 1}, # Insert at the beginning of the document
                    "text": text_to_append,
                }
            }
        ]
        service.documents().batchUpdate(documentId=GOOGLE_DOCS_ID, body={"requests": requests}).execute()
        logger.info("Successfully updated Google Doc.")
    except Exception as e:
        logger.error(f"An error occurred while updating Google Docs: {e}")
        logger.info("The analysis will be printed to console instead:")
        logger.info(text_to_append)

# --- Main Execution ---
# The main execution block is removed to convert this file into a library module.
# The functionality will be invoked from the main server or a dedicated script.