"""
Learning service for the GitHub Issue Solver MCP Server.

Handles storage, retrieval, and semantic search of PR learnings
per repository using JSON files and ChromaDB for embeddings.
"""

import json
import uuid
import re
from loguru import logger
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
from threading import RLock

from langchain_chroma import Chroma

from ..config import Config
from ..exceptions import LearningError
from ..models import (
    RepositoryLearnings,
    LearningPattern,
    NeverDoRule,
    PRTakeaway,
    LearningType,
)
from .embedding_service import EmbeddingService


class LearningService:
    """Manages PR learnings storage, retrieval, and semantic search."""

    def __init__(
        self,
        config: Config,
        embedding_service: EmbeddingService
    ):
        """
        Initialize learning service.

        Args:
            config: Configuration instance
            embedding_service: Embedding service for semantic search
        """
        self.config = config
        self.embedding_service = embedding_service
        self._lock = RLock()

        # Learnings storage directory
        self._learnings_dir = Path(config.chroma_persist_dir) / "learnings"
        self._learnings_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"LearningService initialized. Storage: {self._learnings_dir}")

    def _get_learnings_file(self, repo_name: str) -> Path:
        """Get path to learnings JSON file for a repository."""
        safe_name = repo_name.replace("/", "_").replace("\\", "_")
        return self._learnings_dir / f"{safe_name}.json"

    def _get_collection_name(self, repo_name: str) -> str:
        """Get ChromaDB collection name for learnings."""
        safe_name = repo_name.replace("/", "_").replace("\\", "_").replace("-", "_").lower()
        return f"learnings_{safe_name}"

    def _generate_id(self, prefix: str) -> str:
        """Generate unique ID with prefix."""
        return f"{prefix}-{uuid.uuid4().hex[:8]}"

    # ==================== LOAD/SAVE OPERATIONS ====================

    def load_learnings(self, repo_name: str) -> RepositoryLearnings:
        """
        Load all learnings for a repository.

        Args:
            repo_name: Repository name in 'owner/repo' format

        Returns:
            RepositoryLearnings object (empty if not found)
        """
        with self._lock:
            try:
                learnings_file = self._get_learnings_file(repo_name)

                if not learnings_file.exists():
                    logger.debug(f"No learnings file found for {repo_name}, returning empty")
                    return RepositoryLearnings(
                        repository=repo_name,
                        created_at=datetime.now(),
                        updated_at=datetime.now()
                    )

                with open(learnings_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                learnings = RepositoryLearnings.from_dict(data)
                logger.info(f"Loaded {learnings.get_total_learnings_count()} learnings for {repo_name}")
                return learnings

            except Exception as e:
                logger.error(f"Failed to load learnings for {repo_name}: {e}")
                raise LearningError(
                    f"Failed to load learnings: {e}",
                    repository=repo_name,
                    operation="load"
                )

    def _save_learnings(self, learnings: RepositoryLearnings) -> None:
        """
        Save learnings to JSON file (internal method).

        Args:
            learnings: RepositoryLearnings object to save
        """
        try:
            learnings_file = self._get_learnings_file(learnings.repository)
            learnings.updated_at = datetime.now()

            # Atomic write using temporary file
            temp_file = learnings_file.with_suffix('.tmp')
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(learnings.to_dict(), f, indent=2, ensure_ascii=False)

            # Atomic move
            temp_file.replace(learnings_file)
            logger.debug(f"Saved learnings for {learnings.repository}")

        except Exception as e:
            logger.error(f"Failed to save learnings: {e}")
            raise LearningError(
                f"Failed to save learnings: {e}",
                repository=learnings.repository,
                operation="save"
            )

    # ==================== ADD LEARNING OPERATIONS ====================

    def add_never_do_rule(
        self,
        repo_name: str,
        rule: str,
        reason: str,
        source_pr: Optional[str] = None
    ) -> NeverDoRule:
        """
        Add a "never do" rule for a repository.

        Args:
            repo_name: Repository name
            rule: The rule (what should never be done)
            reason: Why this should never be done
            source_pr: Optional PR reference where this was learned

        Returns:
            Created NeverDoRule
        """
        with self._lock:
            learnings = self.load_learnings(repo_name)

            new_rule = NeverDoRule(
                id=self._generate_id("nd"),
                rule=rule,
                reason=reason,
                source_pr=source_pr,
                added_at=datetime.now()
            )

            learnings.never_do.append(new_rule)
            self._save_learnings(learnings)

            # Add to ChromaDB for semantic search
            self._embed_learning(
                repo_name=repo_name,
                learning_id=new_rule.id,
                learning_type="never_do",
                content=f"NEVER DO: {rule}. Reason: {reason}",
                metadata={"source_pr": source_pr or ""}
            )

            logger.info(f"Added never_do rule '{new_rule.id}' for {repo_name}")
            return new_rule

    def add_pattern(
        self,
        repo_name: str,
        name: str,
        language: str,
        description: str,
        do_example: str,
        dont_example: str,
        tags: Optional[List[str]] = None,
        source_pr: Optional[str] = None
    ) -> LearningPattern:
        """
        Add a code pattern learning.

        Args:
            repo_name: Repository name
            name: Pattern name (e.g., "Performance Guard")
            language: Programming language
            description: What this pattern does
            do_example: Code example of the correct way
            dont_example: Code example of the wrong way
            tags: Optional categorization tags
            source_pr: Optional PR reference

        Returns:
            Created LearningPattern
        """
        with self._lock:
            learnings = self.load_learnings(repo_name)

            new_pattern = LearningPattern(
                id=self._generate_id("pat"),
                name=name,
                language=language,
                description=description,
                do_example=do_example,
                dont_example=dont_example,
                tags=tags or [],
                source_pr=source_pr,
                added_at=datetime.now()
            )

            learnings.patterns.append(new_pattern)
            self._save_learnings(learnings)

            # Add to ChromaDB for semantic search
            self._embed_learning(
                repo_name=repo_name,
                learning_id=new_pattern.id,
                learning_type="pattern",
                content=f"Pattern: {name} ({language}). {description}. DO: {do_example}. DON'T: {dont_example}",
                metadata={
                    "language": language,
                    "tags": ",".join(tags or []),
                    "source_pr": source_pr or ""
                }
            )

            logger.info(f"Added pattern '{new_pattern.id}' ({name}) for {repo_name}")
            return new_pattern

    def add_checklist_item(
        self,
        repo_name: str,
        checklist_name: str,
        item: str
    ) -> None:
        """
        Add an item to a checklist.

        Args:
            repo_name: Repository name
            checklist_name: Name of the checklist (e.g., "backend_rust")
            item: Checklist item text
        """
        with self._lock:
            learnings = self.load_learnings(repo_name)

            if checklist_name not in learnings.checklists:
                learnings.checklists[checklist_name] = []

            if item not in learnings.checklists[checklist_name]:
                learnings.checklists[checklist_name].append(item)
                self._save_learnings(learnings)
                logger.info(f"Added checklist item to '{checklist_name}' for {repo_name}")

    def add_pr_takeaway(
        self,
        repo_name: str,
        original_pr: str,
        title: str,
        result: str,
        lessons: List[Dict[str, str]],
        merged_pr: Optional[str] = None,
        files_touched: Optional[List[str]] = None
    ) -> PRTakeaway:
        """
        Add a PR takeaway learning.

        Args:
            repo_name: Repository name
            original_pr: Original PR number (e.g., "#7329")
            title: Title/description of the PR
            result: What was the outcome
            lessons: List of {"lesson": "...", "action": "..."} dicts
            merged_pr: Final merged PR number if different
            files_touched: List of files that were modified

        Returns:
            Created PRTakeaway
        """
        with self._lock:
            learnings = self.load_learnings(repo_name)

            new_takeaway = PRTakeaway(
                id=self._generate_id("pr"),
                original_pr=original_pr,
                merged_pr=merged_pr,
                title=title,
                result=result,
                lessons=lessons,
                files_touched=files_touched or [],
                added_at=datetime.now()
            )

            learnings.pr_takeaways.append(new_takeaway)
            self._save_learnings(learnings)

            # Add to ChromaDB for semantic search
            lessons_text = " ".join([f"{l['lesson']}: {l['action']}" for l in lessons])
            self._embed_learning(
                repo_name=repo_name,
                learning_id=new_takeaway.id,
                learning_type="pr_takeaway",
                content=f"PR {original_pr}: {title}. Result: {result}. Lessons: {lessons_text}",
                metadata={
                    "original_pr": original_pr,
                    "merged_pr": merged_pr or "",
                    "files": ",".join(files_touched or [])
                }
            )

            logger.info(f"Added PR takeaway '{new_takeaway.id}' for {repo_name}")
            return new_takeaway

    def add_file_group(
        self,
        repo_name: str,
        group_name: str,
        files: List[str]
    ) -> None:
        """
        Add or update a file group.

        Args:
            repo_name: Repository name
            group_name: Name of the file group (e.g., "parser_prs_rust")
            files: List of file paths
        """
        with self._lock:
            learnings = self.load_learnings(repo_name)
            learnings.file_groups[group_name] = files
            self._save_learnings(learnings)
            logger.info(f"Added file group '{group_name}' with {len(files)} files for {repo_name}")

    # ==================== CHROMADB EMBEDDING ====================

    def _embed_learning(
        self,
        repo_name: str,
        learning_id: str,
        learning_type: str,
        content: str,
        metadata: Dict[str, str]
    ) -> None:
        """
        Embed a learning into ChromaDB for semantic search.

        Args:
            repo_name: Repository name
            learning_id: Unique ID of the learning
            learning_type: Type of learning (never_do, pattern, etc.)
            content: Text content to embed
            metadata: Additional metadata
        """
        try:
            collection_name = self._get_collection_name(repo_name)

            # Get embeddings function from service
            embeddings = self.embedding_service.get_embeddings()

            # Create/get the Chroma collection
            store = Chroma(
                collection_name=collection_name,
                embedding_function=embeddings,
                persist_directory=str(self.config.chroma_persist_dir)
            )

            # Add the document
            store.add_texts(
                texts=[content],
                metadatas=[{
                    "id": learning_id,
                    "type": learning_type,
                    "repository": repo_name,
                    **metadata
                }],
                ids=[learning_id]
            )

            logger.debug(f"Embedded learning {learning_id} into {collection_name}")

        except Exception as e:
            # Don't fail the whole operation if embedding fails
            logger.warning(f"Failed to embed learning {learning_id}: {e}")

    # ==================== SEARCH OPERATIONS ====================

    def search_learnings(
        self,
        repo_name: str,
        query: str,
        n_results: int = 5,
        learning_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search learnings using semantic similarity.

        Args:
            repo_name: Repository name
            query: Natural language search query
            n_results: Number of results to return
            learning_type: Optional filter by type

        Returns:
            List of matching learnings with scores
        """
        try:
            collection_name = self._get_collection_name(repo_name)

            # Get embeddings function from service
            embeddings = self.embedding_service.get_embeddings()

            try:
                store = Chroma(
                    collection_name=collection_name,
                    embedding_function=embeddings,
                    persist_directory=str(self.config.chroma_persist_dir)
                )
            except Exception:
                logger.debug(f"No learnings collection found for {repo_name}")
                return []

            # Build filter
            filter_dict = None
            if learning_type:
                filter_dict = {"type": learning_type}

            # Search
            results = store.similarity_search_with_relevance_scores(
                query,
                k=n_results,
                filter=filter_dict
            )

            # Format results
            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    "id": doc.metadata.get("id", ""),
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "relevance_score": score,
                })

            logger.info(f"Found {len(formatted_results)} learnings for query in {repo_name}")
            return formatted_results

        except Exception as e:
            logger.error(f"Failed to search learnings: {e}")
            raise LearningError(
                f"Failed to search learnings: {e}",
                repository=repo_name,
                operation="search"
            )

    # ==================== PRE-PR CHECKLIST ====================

    def get_pre_pr_checklist(
        self,
        repo_name: str,
        files_changed: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get relevant checklist and warnings before making a PR.

        Args:
            repo_name: Repository name
            files_changed: Optional list of files being modified

        Returns:
            Dict with never_do warnings, applicable checklists, and relevant patterns
        """
        learnings = self.load_learnings(repo_name)

        result = {
            "repository": repo_name,
            "never_do_warnings": [rule.to_dict() for rule in learnings.never_do],
            "applicable_checklists": {},
            "relevant_patterns": [],
            "file_groups_matched": [],
        }

        # Determine applicable checklists based on file extensions
        if files_changed:
            extensions = set()
            for f in files_changed:
                if '.' in f:
                    ext = f.rsplit('.', 1)[-1].lower()
                    extensions.add(ext)

            # Map extensions to checklist names
            ext_to_checklist = {
                'rs': ['backend_rust', 'rust'],
                'py': ['backend_python', 'python'],
                'ts': ['frontend_typescript', 'typescript', 'cli_utils'],
                'tsx': ['frontend_typescript', 'typescript'],
                'svelte': ['frontend_svelte', 'svelte'],
                'js': ['frontend_javascript', 'javascript'],
                'sql': ['database', 'migrations'],
            }

            for ext in extensions:
                if ext in ext_to_checklist:
                    for checklist_name in ext_to_checklist[ext]:
                        if checklist_name in learnings.checklists:
                            result["applicable_checklists"][checklist_name] = learnings.checklists[checklist_name]

            # Check file groups
            for group_name, group_files in learnings.file_groups.items():
                for changed_file in files_changed:
                    for group_file in group_files:
                        # Simple pattern matching
                        if group_file.endswith('*'):
                            if changed_file.startswith(group_file[:-1]):
                                result["file_groups_matched"].append(group_name)
                                break
                        elif changed_file == group_file or changed_file.endswith(group_file):
                            result["file_groups_matched"].append(group_name)
                            break

        # Get all checklists if no files specified
        if not files_changed:
            result["applicable_checklists"] = learnings.checklists

        # Get relevant patterns (limit to recent/important ones)
        result["relevant_patterns"] = [p.to_dict() for p in learnings.patterns[:10]]

        return result

    # ==================== MARKDOWN IMPORT ====================

    def import_from_markdown(
        self,
        repo_name: str,
        markdown_path: str
    ) -> Dict[str, int]:
        """
        Import learnings from a markdown file.

        Supports the format used in WINDMILL_PR_LEARNINGS.md:
        - ## NEVER DO section
        - ## Pre-PR Checklist section
        - ## Learned Patterns section
        - ## PR Takeaways section

        Args:
            repo_name: Repository name to store learnings under
            markdown_path: Path to the markdown file

        Returns:
            Dict with counts of imported items
        """
        try:
            with open(markdown_path, 'r', encoding='utf-8') as f:
                content = f.read()

            counts = {
                "never_do": 0,
                "patterns": 0,
                "checklists": 0,
                "pr_takeaways": 0,
            }

            learnings = self.load_learnings(repo_name)

            # Parse NEVER DO section
            never_do_match = re.search(
                r'##\s*[^\n]*NEVER\s*DO[^\n]*\n(.*?)(?=\n##|\Z)',
                content,
                re.IGNORECASE | re.DOTALL
            )
            if never_do_match:
                never_do_content = never_do_match.group(1)
                # Find numbered items
                items = re.findall(r'\d+\.\s*\*\*([^*]+)\*\*\s*[-–]\s*(.+?)(?=\n\d+\.|\n\n|\Z)', never_do_content, re.DOTALL)
                for rule, reason in items:
                    rule = rule.strip()
                    reason = reason.strip()
                    if rule and reason:
                        new_rule = NeverDoRule(
                            id=self._generate_id("nd"),
                            rule=rule,
                            reason=reason,
                            added_at=datetime.now()
                        )
                        learnings.never_do.append(new_rule)
                        counts["never_do"] += 1

            # Parse Pre-PR Checklist section
            checklist_match = re.search(
                r'##\s*Pre-PR\s*Checklist[^\n]*\n(.*?)(?=\n##|\Z)',
                content,
                re.IGNORECASE | re.DOTALL
            )
            if checklist_match:
                checklist_content = checklist_match.group(1)
                # Find subsections like ### Backend (Rust)
                subsections = re.findall(
                    r'###\s*([^\n]+)\n(.*?)(?=\n###|\n##|\Z)',
                    checklist_content,
                    re.DOTALL
                )
                for section_name, section_content in subsections:
                    # Convert "Backend (Rust)" to "backend_rust"
                    checklist_name = re.sub(r'[^a-zA-Z0-9]+', '_', section_name.lower()).strip('_')
                    # Find checkbox items
                    items = re.findall(r'-\s*\[.\]\s*(.+)', section_content)
                    if items:
                        learnings.checklists[checklist_name] = [item.strip() for item in items]
                        counts["checklists"] += len(items)

            # Parse Learned Patterns section
            patterns_match = re.search(
                r'##\s*Learned\s*Patterns[^\n]*\n(.*?)(?=\n##\s*PR\s*Takeaways|\n##\s*Files|\Z)',
                content,
                re.IGNORECASE | re.DOTALL
            )
            if patterns_match:
                patterns_content = patterns_match.group(1)
                # Find pattern subsections like ### 1. Performance Guard (Rust)
                pattern_sections = re.findall(
                    r'###\s*\d+\.\s*([^\n(]+)(?:\(([^)]+)\))?\n(.*?)(?=\n###\s*\d+\.|\n##|\Z)',
                    patterns_content,
                    re.DOTALL
                )
                for name, language, pattern_content in pattern_sections:
                    name = name.strip()
                    language = (language or "general").strip().lower()

                    # Extract DO and DON'T examples from code blocks
                    do_match = re.search(r'//\s*DO[^\n]*\n```[^\n]*\n(.*?)```', pattern_content, re.DOTALL)
                    dont_match = re.search(r"//\s*DON'?T[^\n]*\n```[^\n]*\n(.*?)```", pattern_content, re.DOTALL)

                    do_example = do_match.group(1).strip() if do_match else ""
                    dont_example = dont_match.group(1).strip() if dont_match else ""

                    # Get description (text before first code block)
                    desc_match = re.search(r'^(.*?)```', pattern_content, re.DOTALL)
                    description = desc_match.group(1).strip() if desc_match else ""

                    if name:
                        new_pattern = LearningPattern(
                            id=self._generate_id("pat"),
                            name=name,
                            language=language,
                            description=description,
                            do_example=do_example,
                            dont_example=dont_example,
                            tags=[language],
                            added_at=datetime.now()
                        )
                        learnings.patterns.append(new_pattern)
                        counts["patterns"] += 1

            # Parse PR Takeaways section
            takeaways_match = re.search(
                r'##\s*PR\s*Takeaways[^\n]*\n(.*?)(?=\n##\s*Files|\Z)',
                content,
                re.IGNORECASE | re.DOTALL
            )
            if takeaways_match:
                takeaways_content = takeaways_match.group(1)
                # Find takeaway subsections like ### #7282 -> #7306 (MCP Wildcard UI)
                takeaway_sections = re.findall(
                    r'###\s*#?(\d+)\s*(?:->|→)\s*#?(\d+)\s*\(([^)]+)\)\s*\n(.*?)(?=\n###|\n##|\Z)',
                    takeaways_content,
                    re.DOTALL
                )
                for original_pr, merged_pr, title, takeaway_content in takeaway_sections:
                    # Find Result line
                    result_match = re.search(r'\*\*Result:\*\*\s*(.+)', takeaway_content)
                    result = result_match.group(1).strip() if result_match else ""

                    # Find lessons from table
                    lessons = []
                    lesson_rows = re.findall(r'\|\s*([^|]+)\s*\|\s*([^|]+)\s*\|', takeaway_content)
                    for lesson, action in lesson_rows:
                        lesson = lesson.strip()
                        action = action.strip()
                        if lesson and action and lesson != "Lesson" and lesson != "---":
                            lessons.append({"lesson": lesson, "action": action})

                    if original_pr:
                        new_takeaway = PRTakeaway(
                            id=self._generate_id("pr"),
                            original_pr=f"#{original_pr}",
                            merged_pr=f"#{merged_pr}" if merged_pr else None,
                            title=title.strip(),
                            result=result,
                            lessons=lessons,
                            added_at=datetime.now()
                        )
                        learnings.pr_takeaways.append(new_takeaway)
                        counts["pr_takeaways"] += 1

            # Save all learnings
            self._save_learnings(learnings)

            # Embed all learnings for semantic search
            self._embed_all_learnings(learnings)

            logger.info(f"Imported learnings for {repo_name}: {counts}")
            return counts

        except FileNotFoundError:
            raise LearningError(
                f"Markdown file not found: {markdown_path}",
                repository=repo_name,
                operation="import"
            )
        except Exception as e:
            logger.error(f"Failed to import from markdown: {e}")
            raise LearningError(
                f"Failed to import from markdown: {e}",
                repository=repo_name,
                operation="import"
            )

    def _embed_all_learnings(self, learnings: RepositoryLearnings) -> None:
        """Embed all learnings in a RepositoryLearnings object."""
        repo_name = learnings.repository

        for rule in learnings.never_do:
            self._embed_learning(
                repo_name=repo_name,
                learning_id=rule.id,
                learning_type="never_do",
                content=f"NEVER DO: {rule.rule}. Reason: {rule.reason}",
                metadata={"source_pr": rule.source_pr or ""}
            )

        for pattern in learnings.patterns:
            self._embed_learning(
                repo_name=repo_name,
                learning_id=pattern.id,
                learning_type="pattern",
                content=f"Pattern: {pattern.name} ({pattern.language}). {pattern.description}",
                metadata={
                    "language": pattern.language,
                    "tags": ",".join(pattern.tags),
                    "source_pr": pattern.source_pr or ""
                }
            )

        for takeaway in learnings.pr_takeaways:
            lessons_text = " ".join([f"{l['lesson']}: {l['action']}" for l in takeaway.lessons])
            self._embed_learning(
                repo_name=repo_name,
                learning_id=takeaway.id,
                learning_type="pr_takeaway",
                content=f"PR {takeaway.original_pr}: {takeaway.title}. {takeaway.result}. {lessons_text}",
                metadata={
                    "original_pr": takeaway.original_pr,
                    "merged_pr": takeaway.merged_pr or ""
                }
            )

    # ==================== LIST/DELETE OPERATIONS ====================

    def list_repositories_with_learnings(self) -> List[str]:
        """
        List all repositories that have learnings stored.

        Returns:
            List of repository names
        """
        repos = []
        for json_file in self._learnings_dir.glob("*.json"):
            # Convert filename back to repo name
            repo_name = json_file.stem.replace("_", "/", 1)
            repos.append(repo_name)
        return repos

    def delete_learnings(self, repo_name: str) -> bool:
        """
        Delete all learnings for a repository.

        Args:
            repo_name: Repository name

        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            learnings_file = self._get_learnings_file(repo_name)

            if learnings_file.exists():
                learnings_file.unlink()
                logger.info(f"Deleted learnings for {repo_name}")

                # Also try to delete ChromaDB collection
                try:
                    collection_name = self._get_collection_name(repo_name)
                    # Note: Deleting Chroma collections requires direct client access
                    # For now, just delete the JSON file
                except Exception:
                    pass  # Collection may not exist

                return True

            return False
