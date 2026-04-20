#!/usr/bin/env python3
"""
Generate professional PDF evaluation report from RAG eval results.

Usage:
    python -m evals.generate_report                          # default input/output
    python -m evals.generate_report --input evals/report.json --output evals/RAG_Evaluation_Report.pdf
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from fpdf import FPDF

PROJECT_ROOT = Path(__file__).parent.parent


# ── Color palette ──────────────────────────────────────────────
_BG       = (18, 18, 24)       # dark bg
_CARD     = (30, 30, 46)       # card bg
_ACCENT   = (78, 201, 176)     # teal
_TEXT     = (205, 214, 244)    # light text
_DIM      = (88, 91, 112)     # dim text
_OK       = (166, 227, 161)   # green
_WARN     = (249, 226, 175)   # yellow
_ERR      = (243, 139, 168)   # red
_WHITE    = (255, 255, 255)
_BLACK    = (0, 0, 0)

# Professional palette for PDF (dark mode doesn't work well in print)
_P_BG     = (255, 255, 255)
_P_HEAD   = (30, 41, 59)      # slate-800
_P_ACC    = (14, 116, 144)    # cyan-700
_P_ACC2   = (21, 128, 61)     # green-700
_P_LIGHT  = (241, 245, 249)   # slate-100
_P_BORDER = (203, 213, 225)   # slate-300
_P_TEXT   = (30, 41, 59)      # slate-800
_P_DIM    = (100, 116, 139)   # slate-500


class EvalReport(FPDF):
    """Professional RAG evaluation PDF report."""

    def __init__(self, data: dict):
        super().__init__()
        self.data = data
        self.set_auto_page_break(auto=True, margin=20)

    def header(self):
        if self.page_no() == 1:
            return  # custom header on page 1
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(*_P_DIM)
        self.cell(0, 8, "GIS RAG Evaluation Report", align="L")
        self.cell(0, 8, f"Page {self.page_no()}", align="R", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*_P_BORDER)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*_P_DIM)
        self.cell(0, 10, f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} | GIS v4.0", align="C")

    # ── Helpers ────────────────────────────────────────────────

    def _section(self, title: str):
        self.ln(4)
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(*_P_HEAD)
        self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*_P_ACC)
        self.set_line_width(0.6)
        self.line(10, self.get_y(), 80, self.get_y())
        self.set_line_width(0.2)
        self.ln(4)

    def _subsection(self, title: str):
        self.ln(2)
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(*_P_HEAD)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def _text(self, text: str, size: int = 10):
        self.set_font("Helvetica", "", size)
        self.set_text_color(*_P_TEXT)
        self.set_x(10)
        self.multi_cell(0, 5.5, text)

    def _bold_text(self, text: str, size: int = 10):
        self.set_font("Helvetica", "B", size)
        self.set_text_color(*_P_TEXT)
        self.set_x(10)
        self.multi_cell(0, 5.5, text)

    def _metric_row(self, label: str, value: float, description: str):
        y = self.get_y()
        bar_max_w = 55  # max bar width at value=1.0

        # Background stripe
        self.set_fill_color(*_P_LIGHT)
        self.rect(10, y, 190, 14, "F")

        # Label (left)
        self.set_xy(12, y + 2)
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(*_P_HEAD)
        self.cell(55, 10, label)

        # Score bar
        bar_x = 70
        bar_w = value * bar_max_w
        if value >= 0.8:
            self.set_fill_color(*_P_ACC2)
        elif value >= 0.6:
            self.set_fill_color(202, 138, 4)  # amber
        else:
            self.set_fill_color(220, 38, 38)  # red
        self.rect(bar_x, y + 3, bar_w, 8, "F")

        # Score value (right of bar area)
        self.set_xy(bar_x + bar_max_w + 4, y + 2)
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(*_P_HEAD)
        self.cell(18, 10, f"{value:.2f}")

        # Percentage
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*_P_DIM)
        self.cell(18, 10, f"({value:.0%})")

        # Description (far right)
        self.set_xy(160, y + 2)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*_P_DIM)
        self.cell(40, 10, description)

        self.set_y(y + 16)

    def _table_header(self, cols: list[tuple[str, int]]):
        self.set_fill_color(*_P_HEAD)
        self.set_text_color(*_WHITE)
        self.set_font("Helvetica", "B", 9)
        for label, width in cols:
            self.cell(width, 8, label, border=0, fill=True, align="C")
        self.ln()

    def _table_row(self, values: list[str], widths: list[int], alt: bool = False):
        if alt:
            self.set_fill_color(*_P_LIGHT)
        else:
            self.set_fill_color(*_P_BG)
        self.set_text_color(*_P_TEXT)
        self.set_font("Helvetica", "", 8)
        for val, w in zip(values, widths):
            self.cell(w, 7, val, border=0, fill=True, align="C")
        self.ln()

    # ── Pages ──────────────────────────────────────────────────

    def build(self):
        self._page_cover()
        self._page_methodology()
        self._page_results()
        self._page_detailed()
        self._page_config()

    def _page_cover(self):
        self.add_page()
        self.ln(30)

        # Title
        self.set_font("Helvetica", "B", 28)
        self.set_text_color(*_P_HEAD)
        self.cell(0, 15, "RAG Evaluation Report", align="C", new_x="LMARGIN", new_y="NEXT")

        self.ln(4)
        self.set_font("Helvetica", "", 14)
        self.set_text_color(*_P_ACC)
        self.cell(0, 10, "GIS - GitHub Issue Solver", align="C", new_x="LMARGIN", new_y="NEXT")

        self.ln(8)
        self.set_font("Helvetica", "", 11)
        self.set_text_color(*_P_DIM)
        ts = self.data.get("timestamp", "")[:10]
        self.cell(0, 8, f"Evaluation Date: {ts}", align="C", new_x="LMARGIN", new_y="NEXT")

        cfg = self.data.get("config", {})
        self.cell(0, 8,
                  f"LLM: {cfg.get('llm_provider', '?')} / {cfg.get('llm_model', '?')}",
                  align="C", new_x="LMARGIN", new_y="NEXT")
        self.cell(0, 8,
                  f"Embeddings: {cfg.get('embedding_provider', '?')} ({cfg.get('embedding_model', '?')})",
                  align="C", new_x="LMARGIN", new_y="NEXT")

        # Overall score box
        self.ln(15)
        summary = self.data.get("summary", {})
        overall = summary.get("overall_score", 0)

        box_x = 65
        box_w = 80
        box_y = self.get_y()
        self.set_fill_color(*_P_LIGHT)
        self.set_draw_color(*_P_ACC)
        self.set_line_width(1)
        self.rect(box_x, box_y, box_w, 40, "FD")
        self.set_line_width(0.2)

        self.set_xy(box_x, box_y + 5)
        self.set_font("Helvetica", "", 11)
        self.set_text_color(*_P_DIM)
        self.cell(box_w, 8, "OVERALL SCORE", align="C", new_x="LMARGIN", new_y="NEXT")

        self.set_xy(box_x, box_y + 16)
        self.set_font("Helvetica", "B", 32)
        if overall >= 0.8:
            self.set_text_color(*_P_ACC2)
        elif overall >= 0.6:
            self.set_text_color(202, 138, 4)
        else:
            self.set_text_color(220, 38, 38)
        self.cell(box_w, 18, f"{overall:.1%}", align="C")

        self.set_y(box_y + 50)
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*_P_DIM)
        n = summary.get("scored_questions", 0)
        self.cell(0, 8, f"Based on {n} evaluation questions", align="C", new_x="LMARGIN", new_y="NEXT")

    def _page_methodology(self):
        self.add_page()
        self._section("Evaluation Methodology")

        self._text(
            "This report evaluates the Retrieval-Augmented Generation (RAG) pipeline of the "
            "GIS (GitHub Issue Solver) system. The evaluation measures how effectively the system "
            "retrieves relevant code context from its vector database and generates accurate answers."
        )
        self.ln(3)

        self._subsection("Pipeline Under Test")
        self._text(
            "1. Query Embedding: User question is embedded using the configured embedding model.\n"
            "2. Vector Retrieval: Top-k most similar chunks are retrieved from ChromaDB.\n"
            "3. Context Assembly: Retrieved chunks are assembled into a context window.\n"
            "4. LLM Generation: The LLM generates an answer using only the retrieved context.\n"
            "5. Answer Delivery: The generated answer is returned to the user."
        )
        self.ln(3)

        self._subsection("Evaluation Approach: LLM-as-Judge")
        self._text(
            "Each question is scored by the same LLM acting as an impartial judge. "
            "The judge evaluates four orthogonal quality dimensions, each scored 0.0 to 1.0:"
        )
        self.ln(3)

        metrics_desc = [
            ("Context Precision", "What fraction of retrieved chunks are actually relevant to the query? "
             "High precision means the retrieval avoids returning irrelevant noise."),
            ("Context Recall", "Did the retrieval find all the information needed to answer correctly? "
             "Measured against the ground-truth answer. High recall means no critical context is missed."),
            ("Faithfulness", "Does the generated answer stick strictly to the retrieved context without "
             "hallucinating facts? A score of 1.0 means zero hallucination."),
            ("Answer Relevancy", "Does the final answer actually address the question asked? "
             "Measured against both the question and the expected ground-truth answer."),
        ]

        for name, desc in metrics_desc:
            self._bold_text(name)
            self._text(desc)
            self.ln(1)

        self.ln(3)
        self._subsection("Ground-Truth Dataset")
        self._text(
            f"The evaluation uses {self.data['summary']['total_questions']} curated question-answer pairs "
            "targeting different aspects of the ingested repository. Each question has a manually written "
            "ground-truth answer that serves as the reference for recall and relevancy scoring. "
            "Questions span code architecture, configuration, APIs, issues, and documentation."
        )

    def _page_results(self):
        self.add_page()
        self._section("Results Summary")

        summary = self.data.get("summary", {})

        # Metric bars
        self._metric_row("Context Precision", summary.get("context_precision", 0),
                         "Retrieval relevance")
        self._metric_row("Context Recall", summary.get("context_recall", 0),
                         "Retrieval completeness")
        self._metric_row("Faithfulness", summary.get("faithfulness", 0),
                         "Hallucination control")
        self._metric_row("Answer Relevancy", summary.get("answer_relevancy", 0),
                         "Response quality")

        self.ln(6)

        # Performance stats
        self._subsection("Performance")

        perf_data = [
            ("Average Retrieval Latency", f"{summary.get('avg_retrieval_time_s', 0):.3f}s"),
            ("Average Generation Latency", f"{summary.get('avg_generation_time_s', 0):.1f}s"),
            ("Questions Evaluated", str(summary.get("scored_questions", 0))),
            ("Questions Skipped", str(summary.get("skipped", 0))),
        ]

        for label, val in perf_data:
            self.set_font("Helvetica", "", 10)
            self.set_text_color(*_P_TEXT)
            self.cell(80, 7, label)
            self.set_font("Helvetica", "B", 10)
            self.cell(40, 7, val, new_x="LMARGIN", new_y="NEXT")

        self.ln(6)

        # Interpretation
        self._subsection("Interpretation")

        overall = summary.get("overall_score", 0)
        faith = summary.get("faithfulness", 0)
        precision = summary.get("context_precision", 0)
        recall = summary.get("context_recall", 0)
        relevancy = summary.get("answer_relevancy", 0)

        findings = []
        if faith >= 0.95:
            findings.append(
                f"Faithfulness score of {faith:.0%} indicates near-zero hallucination. "
                "The LLM reliably stays within the bounds of retrieved context."
            )
        if precision >= 0.75:
            findings.append(
                f"Context precision of {precision:.0%} shows the retrieval pipeline "
                "returns mostly relevant chunks with minimal noise."
            )
        if recall >= 0.75:
            findings.append(
                f"Context recall of {recall:.0%} demonstrates that the system "
                "finds the majority of information needed to answer correctly."
            )
        if relevancy < 0.75:
            findings.append(
                f"Answer relevancy of {relevancy:.0%} suggests room for improvement in "
                "how well the generated answers directly address the specific question asked. "
                "This can be improved with better prompt engineering or answer post-processing."
            )
        if overall >= 0.8:
            findings.append(
                f"The overall score of {overall:.1%} indicates a production-ready RAG pipeline "
                "with strong retrieval quality and minimal hallucination."
            )

        for f in findings:
            self._text(f"  {f}")
            self.ln(2)

    def _page_detailed(self):
        self.add_page()
        self._section("Per-Question Results")

        results = self.data.get("results", [])

        cols = [
            ("Q#", 10), ("Precision", 22), ("Recall", 22),
            ("Faithful", 22), ("Relevancy", 22), ("Avg", 18),
            ("Retrieval", 22), ("Gen Time", 22), ("Chunks", 16),
        ]
        widths = [c[1] for c in cols]

        # Table header
        self._table_header(cols)

        # Rows
        for i, r in enumerate(results):
            if r.get("skipped"):
                vals = [str(i + 1), "-", "-", "-", "-", "-", "-", "-", "-"]
            else:
                m = r.get("metrics", {})
                vals = [
                    str(i + 1),
                    f"{m.get('context_precision', 0):.2f}",
                    f"{m.get('context_recall', 0):.2f}",
                    f"{m.get('faithfulness', 0):.2f}",
                    f"{m.get('answer_relevancy', 0):.2f}",
                    f"{r.get('avg_score', 0):.2f}",
                    f"{r.get('retrieval_time_s', 0):.3f}s",
                    f"{r.get('generation_time_s', 0):.1f}s",
                    str(r.get("chunks_retrieved", 0)),
                ]
            self._table_row(vals, widths, alt=(i % 2 == 1))

        self.ln(8)

        # Detailed Q&A
        self._subsection("Question Details")

        for i, r in enumerate(results):
            if r.get("skipped"):
                continue

            if self.get_y() > 230:
                self.add_page()

            self.set_font("Helvetica", "B", 9)
            self.set_text_color(*_P_ACC)
            self.cell(0, 6, f"Q{i + 1}: {r['question'][:100]}", new_x="LMARGIN", new_y="NEXT")

            self.set_font("Helvetica", "", 8)
            self.set_text_color(*_P_DIM)

            # Sources
            sources = r.get("sources", [])
            if sources:
                src_text = "Sources: " + ", ".join(sources[:3])
                self.cell(0, 5, src_text, new_x="LMARGIN", new_y="NEXT")

            # Scores inline
            m = r.get("metrics", {})
            score_text = (
                f"P={m.get('context_precision', 0):.2f}  "
                f"R={m.get('context_recall', 0):.2f}  "
                f"F={m.get('faithfulness', 0):.2f}  "
                f"A={m.get('answer_relevancy', 0):.2f}  "
                f"Avg={r.get('avg_score', 0):.2f}"
            )
            self.set_font("Helvetica", "B", 8)
            self.set_text_color(*_P_TEXT)
            self.cell(0, 5, score_text, new_x="LMARGIN", new_y="NEXT")

            # Answer preview
            answer = r.get("answer", "")[:200]
            if len(r.get("answer", "")) > 200:
                answer += "..."
            self.set_font("Helvetica", "", 8)
            self.set_text_color(*_P_DIM)
            self.multi_cell(0, 4, f"Answer: {answer}")
            self.ln(3)

    def _page_config(self):
        self.add_page()
        self._section("Evaluation Configuration")

        cfg = self.data.get("config", {})

        config_items = [
            ("LLM Provider", cfg.get("llm_provider", "?")),
            ("LLM Model", cfg.get("llm_model", "?")),
            ("Embedding Provider", cfg.get("embedding_provider", "?")),
            ("Embedding Model", cfg.get("embedding_model", "?")),
            ("Top-K Retrieval", str(cfg.get("top_k", "?"))),
            ("Vector Database", "ChromaDB (persistent)"),
            ("Evaluation Method", "LLM-as-Judge (same model)"),
            ("Scoring Range", "0.0 - 1.0 per metric"),
        ]

        for label, val in config_items:
            self.set_font("Helvetica", "B", 10)
            self.set_text_color(*_P_TEXT)
            self.cell(65, 7, label)
            self.set_font("Helvetica", "", 10)
            self.set_text_color(*_P_DIM)
            self.cell(0, 7, val, new_x="LMARGIN", new_y="NEXT")

        self.ln(8)
        self._subsection("System Architecture")
        self._text(
            "The GIS RAG pipeline operates in three stages:\n\n"
            "Ingestion: Repositories are ingested in 4 steps (docs, code, issues, PRs). "
            "Content is chunked using RecursiveCharacterTextSplitter with provider-aware "
            "chunk sizes (8-10KB for FastEmbed, 4-6KB for Google embeddings). "
            "Chunks are embedded and stored in isolated ChromaDB collections per repository.\n\n"
            "Retrieval: Queries are embedded using the same model and matched against "
            "stored vectors via cosine similarity. Top-k chunks (default k=5) are returned "
            "with relevance scores.\n\n"
            "Generation: Retrieved chunks are assembled into a context window and passed "
            "to the LLM with a system prompt constraining the answer to the provided context. "
            "This reduces hallucination and ensures grounded responses."
        )

        self.ln(6)
        self._subsection("LLM Provider Support")
        self._text(
            "GIS uses LiteLLM for unified LLM routing, supporting:\n"
            "- Gemini (Google) - gemini-2.5-flash, gemini-2.5-pro\n"
            "- Claude (Anthropic) - claude-sonnet-4-5-20241022\n"
            "- Grok (xAI) - grok-3, grok-3-mini\n"
            "- OpenAI - gpt-4o, gpt-4o-mini\n"
            "- Ollama (local) - llama3.1, mistral, and other local models\n\n"
            "All providers are accessed through a single PROVIDERS registry with "
            "automatic API key validation and LangChain-compatible tool calling."
        )

        self.ln(6)
        self._subsection("Reproducibility")
        self._text(
            "To reproduce this evaluation:\n\n"
            "  pip install -e .\n"
            "  python -m evals.run_eval --output evals/report.json\n"
            "  python -m evals.generate_report --input evals/report.json\n\n"
            "The golden dataset is stored in evals/golden_dataset.json and can be "
            "extended with additional question-answer pairs for more comprehensive coverage."
        )


def generate_pdf_report(input_path: str = None, output_path: str = None) -> str:
    """Generate PDF report from eval JSON. Returns output path."""
    if input_path is None:
        input_path = str(PROJECT_ROOT / "evals" / "report.json")
    if output_path is None:
        output_path = str(PROJECT_ROOT / "evals" / "RAG_Evaluation_Report.pdf")

    with open(input_path) as f:
        data = json.load(f)

    report = EvalReport(data)
    report.build()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    report.output(output_path)
    print(f"  Report saved: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate RAG eval PDF report")
    parser.add_argument("--input", default=None, help="Input JSON report")
    parser.add_argument("--output", default=None, help="Output PDF path")
    args = parser.parse_args()
    generate_pdf_report(input_path=args.input, output_path=args.output)


if __name__ == "__main__":
    main()
