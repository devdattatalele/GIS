#!/usr/bin/env python3
"""
GIS RAG Evaluation Runner.

Measures retrieval quality of the RAG pipeline using:
  - Context Precision: are retrieved chunks relevant?
  - Context Recall:    did we find all relevant chunks?
  - Faithfulness:      does the LLM answer stick to context?
  - Answer Relevancy:  does the answer address the question?

Usage:
    python -m evals.run_eval                           # run full eval
    python -m evals.run_eval --repo windmill-labs/windmill
    python -m evals.run_eval --embedding fastembed     # compare providers
    python -m evals.run_eval --output evals/report.json
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def load_golden_dataset(path: str = None) -> list[dict]:
    """Load ground-truth Q&A dataset."""
    if path is None:
        path = str(PROJECT_ROOT / "evals" / "golden_dataset.json")
    with open(path) as f:
        return json.load(f)


def _resolve_collection(repo_name: str, collection_type: str,
                        config, chroma_dir: str) -> str:
    """Find the actual collection name in ChromaDB.

    Handles both user-isolated (v4+) and legacy (v3) naming.
    """
    import chromadb

    # Try the current config naming first
    candidate = config.get_collection_name(repo_name, collection_type)

    client = chromadb.PersistentClient(path=chroma_dir)
    collections = {c.name: c.count() for c in client.list_collections()}

    # Only use candidate if it has data
    if candidate in collections and collections[candidate] > 0:
        return candidate

    # Fallback: legacy naming without user prefix
    collection_map = {
        "documentation": "documentation",
        "code": "repo_code_main",
        "issues": "issues_history",
        "prs": "pr_history",
    }
    safe = repo_name.replace("/", "_").replace("-", "_").lower()
    base = collection_map.get(collection_type, collection_type)
    legacy = f"{safe}_{base}"

    # Try exact legacy name
    if legacy in collections and collections[legacy] > 0:
        return legacy

    # Try partial match — prefer collections with data
    for name, count in sorted(collections.items(), key=lambda x: -x[1]):
        if count > 0 and safe in name and collection_type[:4] in name:
            return name

    return candidate  # return the config name even if not found


def retrieve_chunks(query: str, repo_name: str, collection_type: str,
                    config, embedding_service, k: int = 5) -> list[str]:
    """Retrieve top-k chunks from ChromaDB for a query."""
    from langchain_chroma import Chroma

    chroma_dir = str(config.chroma_persist_dir)
    collection_name = _resolve_collection(repo_name, collection_type,
                                           config, chroma_dir)
    embeddings = embedding_service.get_embeddings()

    try:
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=chroma_dir,
        )
        docs = vectorstore.similarity_search_with_relevance_scores(query, k=k)
        return [
            {"content": doc.page_content, "score": score,
             "source": doc.metadata.get("source", "unknown")}
            for doc, score in docs
        ]
    except Exception as e:
        print(f"  [WARN] retrieval failed for {collection_name}: {e}")
        return []


def generate_answer(query: str, contexts: list[str], llm) -> str:
    """Generate an answer using LLM with retrieved context."""
    from langchain_core.messages import HumanMessage, SystemMessage

    ctx_text = "\n\n---\n\n".join(contexts[:5])
    messages = [
        SystemMessage(content=(
            "You are a precise technical assistant. Answer the question using "
            "ONLY the provided context. If the context doesn't contain enough "
            "information, say so. Be concise."
        )),
        HumanMessage(content=f"Context:\n{ctx_text}\n\nQuestion: {query}"),
    ]

    try:
        response = llm.invoke(messages)
        content = response.content
        if isinstance(content, list):
            content = "\n".join(
                b.get("text", "") if isinstance(b, dict) else str(b)
                for b in content
            )
        return content.strip()
    except Exception as e:
        return f"[LLM error: {e}]"


def compute_metrics(question: str, answer: str, contexts: list[str],
                    ground_truth: str, llm) -> dict:
    """Compute RAG quality metrics using LLM-as-judge."""
    from langchain_core.messages import HumanMessage

    def _score(prompt: str) -> float:
        """Ask LLM to score 0.0-1.0."""
        try:
            resp = llm.invoke([HumanMessage(content=prompt)])
            text = resp.content if isinstance(resp.content, str) else str(resp.content)
            # Extract first float from response
            import re
            match = re.search(r"(\d+\.?\d*)", text)
            if match:
                val = float(match.group(1))
                return min(max(val, 0.0), 1.0)
            return 0.0
        except Exception:
            return 0.0

    ctx_joined = "\n---\n".join(contexts[:5])

    # Context Precision: are retrieved chunks relevant to the question?
    precision = _score(
        f"Rate 0.0 to 1.0: how relevant are these retrieved chunks to the question?\n\n"
        f"Question: {question}\n\nRetrieved chunks:\n{ctx_joined}\n\n"
        f"Reply with ONLY a number between 0.0 and 1.0."
    )

    # Context Recall: do retrieved chunks cover the ground truth?
    recall = _score(
        f"Rate 0.0 to 1.0: do these chunks contain enough information to answer correctly?\n\n"
        f"Expected answer: {ground_truth}\n\nRetrieved chunks:\n{ctx_joined}\n\n"
        f"Reply with ONLY a number between 0.0 and 1.0."
    )

    # Faithfulness: does the answer stick to context (no hallucination)?
    faithfulness = _score(
        f"Rate 0.0 to 1.0: is this answer strictly based on the provided context "
        f"without adding external information?\n\n"
        f"Context:\n{ctx_joined}\n\nAnswer: {answer}\n\n"
        f"Reply with ONLY a number between 0.0 and 1.0."
    )

    # Answer Relevancy: does the answer address the question?
    relevancy = _score(
        f"Rate 0.0 to 1.0: how well does this answer address the question?\n\n"
        f"Question: {question}\n\nAnswer: {answer}\n\n"
        f"Expected: {ground_truth}\n\n"
        f"Reply with ONLY a number between 0.0 and 1.0."
    )

    return {
        "context_precision": precision,
        "context_recall": recall,
        "faithfulness": faithfulness,
        "answer_relevancy": relevancy,
    }


def run_evaluation(
    repo_filter: Optional[str] = None,
    embedding_provider: Optional[str] = None,
    dataset_path: Optional[str] = None,
    output_path: Optional[str] = None,
    k: int = 5,
) -> dict:
    """Run full RAG evaluation pipeline."""

    # Suppress noise
    import warnings
    warnings.filterwarnings("ignore")
    os.environ["GRPC_VERBOSITY"] = "NONE"
    os.environ["ABSL_MIN_LOG_LEVEL"] = "99"
    import logging
    # Suppress litellm verbosity
    os.environ["LITELLM_LOG"] = "ERROR"
    for name in ["absl", "grpc", "google", "urllib3", "httpx", "httpcore",
                  "chromadb", "langchain", "langsmith", "opentelemetry",
                  "fastembed", "onnxruntime", "litellm", "LiteLLM",
                  "litellm.utils", "litellm.llms"]:
        logging.getLogger(name).setLevel(logging.CRITICAL)

    # Kill loguru stderr output
    try:
        from loguru import logger as _loguru
        _loguru.remove()
    except ImportError:
        pass

    if embedding_provider:
        os.environ["EMBEDDING_PROVIDER"] = embedding_provider

    from github_issue_solver.config import Config
    from github_issue_solver.services.embedding_service import EmbeddingService
    from github_issue_solver.services.llm_service import LLMService

    print(f"\n  GIS RAG Evaluation\n  {'=' * 40}")

    # Init
    config = Config()
    emb_service = EmbeddingService(config)
    llm_service = LLMService(config)
    llm = llm_service.get_llm()

    print(f"  LLM:        {llm_service.provider_name} / {llm_service.model_name}")
    print(f"  Embeddings: {config.embedding_provider}")
    print(f"  ChromaDB:   {config.chroma_persist_dir}")

    # Load dataset
    dataset = load_golden_dataset(dataset_path)
    if repo_filter:
        dataset = [d for d in dataset if d["repo"] == repo_filter]

    print(f"  Questions:  {len(dataset)}\n")

    results = []
    totals = {"context_precision": 0, "context_recall": 0,
              "faithfulness": 0, "answer_relevancy": 0}

    for i, item in enumerate(dataset, 1):
        question = item["question"]
        ground_truth = item["ground_truth"]
        repo = item["repo"]
        collection = item.get("collection", "code")

        print(f"  [{i}/{len(dataset)}] {question[:70]}...")

        # Retrieve
        t0 = time.time()
        chunks = retrieve_chunks(question, repo, collection, config, emb_service, k=k)
        retrieval_time = time.time() - t0

        contexts = [c["content"] for c in chunks]
        scores = [c["score"] for c in chunks]

        if not contexts:
            print(f"    -> no chunks retrieved, skipping")
            results.append({
                "question": question, "skipped": True,
                "reason": "no chunks retrieved",
            })
            continue

        # Generate answer
        t1 = time.time()
        answer = generate_answer(question, contexts, llm)
        generation_time = time.time() - t1

        # Score
        metrics = compute_metrics(question, answer, contexts, ground_truth, llm)

        avg = sum(metrics.values()) / len(metrics)
        print(f"    -> precision={metrics['context_precision']:.2f}  "
              f"recall={metrics['context_recall']:.2f}  "
              f"faithful={metrics['faithfulness']:.2f}  "
              f"relevancy={metrics['answer_relevancy']:.2f}  "
              f"avg={avg:.2f}  "
              f"({retrieval_time:.1f}s + {generation_time:.1f}s)")

        for k_name, v in metrics.items():
            totals[k_name] += v

        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "answer": answer,
            "metrics": metrics,
            "avg_score": avg,
            "retrieval_time_s": round(retrieval_time, 3),
            "generation_time_s": round(generation_time, 3),
            "chunks_retrieved": len(chunks),
            "top_chunk_score": round(scores[0], 4) if scores else 0,
            "sources": [c["source"] for c in chunks[:3]],
        })

    # Aggregate
    scored = [r for r in results if not r.get("skipped")]
    n = len(scored)

    if n == 0:
        print("\n  No questions were scored. Check your ChromaDB data.")
        return {"error": "no scored questions"}

    averages = {k: round(v / n, 4) for k, v in totals.items()}
    overall = round(sum(averages.values()) / len(averages), 4)

    report = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "llm_provider": llm_service.provider_name,
            "llm_model": llm_service.model_name,
            "embedding_provider": config.embedding_provider,
            "embedding_model": config.embedding_model_name,
            "top_k": k,
        },
        "summary": {
            "total_questions": len(dataset),
            "scored_questions": n,
            "skipped": len(dataset) - n,
            "overall_score": overall,
            **averages,
            "avg_retrieval_time_s": round(
                sum(r["retrieval_time_s"] for r in scored) / n, 3
            ),
            "avg_generation_time_s": round(
                sum(r["generation_time_s"] for r in scored) / n, 3
            ),
        },
        "results": results,
    }

    # Print summary
    print(f"\n  {'=' * 40}")
    print(f"  RESULTS ({n} questions scored)\n")
    print(f"  Context Precision:  {averages['context_precision']:.2f}")
    print(f"  Context Recall:     {averages['context_recall']:.2f}")
    print(f"  Faithfulness:       {averages['faithfulness']:.2f}")
    print(f"  Answer Relevancy:   {averages['answer_relevancy']:.2f}")
    print(f"  ────────────────────────────")
    print(f"  Overall Score:      {overall:.2f}")
    print(f"  Avg Retrieval:      {report['summary']['avg_retrieval_time_s']:.3f}s")
    print(f"  Avg Generation:     {report['summary']['avg_generation_time_s']:.3f}s")

    # Save
    if output_path is None:
        output_path = str(PROJECT_ROOT / "evals" / "report.json")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved: {output_path}\n")

    return report


def main():
    parser = argparse.ArgumentParser(description="GIS RAG Evaluation")
    parser.add_argument("--repo", default=None, help="Filter to specific repo")
    parser.add_argument("--embedding", default=None,
                        help="Override embedding provider (fastembed/google)")
    parser.add_argument("--dataset", default=None, help="Path to golden dataset")
    parser.add_argument("--output", default=None, help="Output report path")
    parser.add_argument("--top-k", type=int, default=5, help="Chunks to retrieve")
    args = parser.parse_args()

    run_evaluation(
        repo_filter=args.repo,
        embedding_provider=args.embedding,
        dataset_path=args.dataset,
        output_path=args.output,
        k=args.top_k,
    )


if __name__ == "__main__":
    main()
