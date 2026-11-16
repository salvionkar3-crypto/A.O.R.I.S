"""Production built-in tools for JARVIS."""

import ast
from datetime import datetime, timezone
import math
import operator
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Union

from jarvis.config import get_settings
from jarvis.core.enums import FactStatus, MemoryCategory
from jarvis.memory.models import MemoryItem
from jarvis.memory.service import MemoryService
from jarvis.tools.base import BaseTool, ToolPermission, ToolResult



class FileReaderTool(BaseTool):
    """Safely reads file contents within permitted directories with size and line limits."""

    @property
    def name(self) -> str:
        return "file_reader"

    @property
    def description(self) -> str:
        return "Read text from a file within permitted workspace directories."

    @property
    def permission(self) -> ToolPermission:
        return ToolPermission.READ_ONLY

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Relative or absolute path to the file"},
                "max_lines": {"type": "integer", "description": "Maximum lines to read", "default": 200},
            },
            "required": ["file_path"],
        }

    def execute(self, file_path: str, max_lines: int = 200, **kwargs: Any) -> ToolResult:
        p = Path(file_path).resolve()
        if not p.exists():
            return ToolResult(tool_name=self.name, success=False, error=f"File '{file_path}' does not exist.")
        if not p.is_file():
            return ToolResult(tool_name=self.name, success=False, error=f"Path '{file_path}' is not a file.")

        try:
            with open(p, "r", encoding="utf-8", errors="replace") as f:
                lines = [f.readline() for _ in range(max_lines)]
                content = "".join(lines)
            return ToolResult(
                tool_name=self.name,
                success=True,
                data={
                    "path": str(p),
                    "lines_read": len(lines),
                    "content": content,
                },
            )
        except Exception as e:
            return ToolResult(tool_name=self.name, success=False, error=f"Failed to read file: {e}")


class FileWriterTool(BaseTool):
    """Writes text content to a file within permitted directories. Requires confirmation."""

    @property
    def name(self) -> str:
        return "file_writer"

    @property
    def description(self) -> str:
        return "Write text to a file within permitted directories. Modifies local storage."

    @property
    def permission(self) -> ToolPermission:
        return ToolPermission.CONSEQUENTIAL

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Path of file to write"},
                "content": {"type": "string", "description": "Text content to write"},
                "append": {"type": "boolean", "description": "Append instead of overwrite", "default": False},
            },
            "required": ["file_path", "content"],
        }

    def preview_action(self, file_path: str, content: str, append: bool = False, **kwargs: Any) -> str:
        mode = "Append to" if append else "Overwrite/Create"
        snippet = content[:150] + ("..." if len(content) > 150 else "")
        return f"{mode} file '{file_path}' with content: {snippet}"

    def execute(self, file_path: str, content: str, append: bool = False, **kwargs: Any) -> ToolResult:
        p = Path(file_path).resolve()
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if append else "w"
            with open(p, mode, encoding="utf-8") as f:
                f.write(content)
            return ToolResult(
                tool_name=self.name,
                success=True,
                data={
                    "path": str(p),
                    "bytes_written": len(content.encode("utf-8")),
                    "mode": mode,
                },
            )
        except Exception as e:
            return ToolResult(tool_name=self.name, success=False, error=f"Failed to write file: {e}")


class FileListTool(BaseTool):
    """Lists directory contents safely."""

    @property
    def name(self) -> str:
        return "file_list"

    @property
    def description(self) -> str:
        return "List files and subdirectories within a permitted directory."

    @property
    def permission(self) -> ToolPermission:
        return ToolPermission.READ_ONLY

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "directory_path": {"type": "string", "description": "Path to directory", "default": "."},
            },
        }

    def execute(self, directory_path: str = ".", **kwargs: Any) -> ToolResult:
        p = Path(directory_path).resolve()
        if not p.exists():
            return ToolResult(tool_name=self.name, success=False, error=f"Directory '{directory_path}' does not exist.")
        if not p.is_dir():
            return ToolResult(tool_name=self.name, success=False, error=f"Path '{directory_path}' is not a directory.")

        try:
            entries = []
            for item in sorted(p.iterdir()):
                entries.append({
                    "name": item.name,
                    "is_dir": item.is_dir(),
                    "size_bytes": item.stat().st_size if item.is_file() else None,
                })
            return ToolResult(tool_name=self.name, success=True, data={"directory": str(p), "entries": entries})
        except Exception as e:
            return ToolResult(tool_name=self.name, success=False, error=f"Failed to list directory: {e}")


# Safe mathematical evaluation AST operators
SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_ast(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body)
    elif isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    elif isinstance(node, ast.UnaryOp):
        op = SAFE_OPERATORS.get(type(node.op))
        if op:
            return op(_eval_ast(node.operand))
    elif isinstance(node, ast.BinOp):
        op = SAFE_OPERATORS.get(type(node.op))
        if op:
            left = _eval_ast(node.left)
            right = _eval_ast(node.right)
            return op(left, right)
    raise ValueError(f"Unsupported or unsafe expression element: {ast.dump(node)}")


class CalculatorTool(BaseTool):
    """Safely evaluates arithmetic expressions using AST parsing without arbitrary code execution."""

    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return "Calculate the mathematical result of an arithmetic expression."

    @property
    def permission(self) -> ToolPermission:
        return ToolPermission.READ_ONLY

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Mathematical formula to evaluate (e.g. 1024 * 768)"},
            },
            "required": ["expression"],
        }

    def execute(self, expression: str, **kwargs: Any) -> ToolResult:
        try:
            tree = ast.parse(expression.strip(), mode="eval")
            result = _eval_ast(tree)
            return ToolResult(
                tool_name=self.name,
                success=True,
                data={"expression": expression, "result": result},
            )
        except Exception as e:
            return ToolResult(tool_name=self.name, success=False, error=f"Calculation error: {e}")


class MemorySearchTool(BaseTool):
    """Searches JARVIS persistent memory service."""

    def __init__(self, memory_service: MemoryService) -> None:
        self.memory_service = memory_service

    @property
    def name(self) -> str:
        return "memory_search"

    @property
    def description(self) -> str:
        return "Search persistent memory for saved facts, preferences, and project context."

    @property
    def permission(self) -> ToolPermission:
        return ToolPermission.READ_ONLY

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search terms or question to match memories against"},
                "limit": {"type": "integer", "description": "Max results to return", "default": 5},
            },
            "required": ["query"],
        }

    def execute(self, query: str, limit: int = 5, **kwargs: Any) -> ToolResult:
        try:
            items = self.memory_service.retrieve_context(query=query, limit=limit)
            return ToolResult(
                tool_name=self.name,
                success=True,
                data={
                    "query": query,
                    "count": len(items),
                    "memories": [
                        {
                            "id": m.id,
                            "content": m.content,
                            "category": m.category.value,
                            "fact_status": m.fact_status.value,
                            "confidence": m.confidence,
                            "created_at": m.created_at.isoformat(),
                        }
                        for m in items
                    ],
                },
            )
        except Exception as e:
            return ToolResult(tool_name=self.name, success=False, error=f"Memory search failed: {e}")


class SystemInfoTool(BaseTool):
    """Provides current time and platform metadata safely without leaking environment secrets."""

    @property
    def name(self) -> str:
        return "system_info"

    @property
    def description(self) -> str:
        return "Get current system time, UTC timestamp, and runtime environment details."

    @property
    def permission(self) -> ToolPermission:
        return ToolPermission.READ_ONLY

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {}}

    def execute(self, **kwargs: Any) -> ToolResult:
        now_utc = datetime.now(timezone.utc)
        return ToolResult(
            tool_name=self.name,
            success=True,
            data={
                "utc_time": now_utc.isoformat(),
                "timestamp": now_utc.timestamp(),
                "system": "JARVIS Autonomous Environment",
                "version": "0.1.0",
            },
        )


class ResearchSearchTool(BaseTool):
    """
    Searches workspace documents, project files, and local documentation for factual evidence with source citations.
    Includes honest fallback if no local documents match and live web search is not configured.
    """

    def __init__(self, workspace_root: Optional[Path] = None) -> None:
        self.workspace_root = workspace_root or Path.cwd()

    @property
    def name(self) -> str:
        return "research_search"

    @property
    def description(self) -> str:
        return (
            "Search local workspace documents, markdown guides, and project context for factual evidence "
            "with source citations. Returns exact quotes, line citations, and honest reporting if unavailable."
        )

    @property
    def permission(self) -> ToolPermission:
        return ToolPermission.READ_ONLY

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search keywords or inquiry to match within documentation"},
                "scope": {"type": "string", "description": "Search scope ('workspace', 'docs', or 'all')", "default": "workspace"},
                "max_results": {"type": "integer", "description": "Maximum number of citations to return", "default": 5},
            },
            "required": ["query"],
        }

    def execute(self, query: str, scope: str = "workspace", max_results: int = 5, **kwargs: Any) -> ToolResult:
        query_terms = [t.lower() for t in query.split() if len(t) > 2]
        if not query_terms:
            query_terms = [query.lower()]

        eligible_extensions = {".md", ".txt", ".rst", ".json", ".py", ".yaml", ".yml"}
        matches: List[Dict[str, Any]] = []
        citations: List[str] = []

        search_dirs = [self.workspace_root]
        docs_dir = self.workspace_root / "docs"
        if docs_dir.exists() and docs_dir.is_dir():
            search_dirs.append(docs_dir)

        try:
            for sdir in search_dirs:
                if not sdir.exists():
                    continue
                for p in sdir.rglob("*"):
                    if not p.is_file() or p.suffix.lower() not in eligible_extensions:
                        continue
                    parts = p.parts
                    if any(ignored in parts for ignored in [".git", ".venv", "venv", "__pycache__", "node_modules", ".pytest_cache"]):
                        continue

                    try:
                        with open(p, "r", encoding="utf-8", errors="replace") as f:
                            for line_no, line in enumerate(f, start=1):
                                line_lower = line.lower()
                                if any(term in line_lower for term in query_terms):
                                    rel_path = p.relative_to(self.workspace_root).as_posix()
                                    snippet = line.strip()
                                    if len(snippet) > 200:
                                        snippet = snippet[:197] + "..."
                                    citation = f"{rel_path}:L{line_no}"
                                    matches.append({
                                        "source": rel_path,
                                        "line": line_no,
                                        "snippet": snippet,
                                        "citation": citation,
                                    })
                                    citations.append(citation)
                                    if len(matches) >= max_results:
                                        break
                    except Exception:
                        continue
                    if len(matches) >= max_results:
                        break

            if matches:
                return ToolResult(
                    tool_name=self.name,
                    success=True,
                    data={
                        "query": query,
                        "matches_found": len(matches),
                        "citations": citations,
                        "results": matches,
                        "live_search_available": False,
                        "note": "Results sourced from verified local repository files.",
                    },
                )
            else:
                return ToolResult(
                    tool_name=self.name,
                    success=True,
                    data={
                        "query": query,
                        "matches_found": 0,
                        "citations": [],
                        "results": [],
                        "live_search_available": False,
                        "note": (
                            f"No matching local workspace files found for '{query}'. "
                            "Live web search provider is currently not configured; "
                            "no synthetic or unverified citations were generated."
                        ),
                    },
                )
        except Exception as e:
            return ToolResult(tool_name=self.name, success=False, error=f"Research search failed: {e}")


class PDFGeneratorTool(BaseTool):
    """
    Generates professional PDF documents, reports, and tables using ReportLab.
    Ensures zero external system dependencies and validates output integrity.
    """

    def __init__(self, output_dir: Optional[Path] = None) -> None:
        self._custom_output_dir = output_dir

    @property
    def output_dir(self) -> Path:
        if self._custom_output_dir:
            return self._custom_output_dir
        return get_settings().aoris_docs_dir

    @property
    def name(self) -> str:
        return "pdf_generator"

    @property
    def description(self) -> str:
        return (
            "Generate professional formatted PDF documents with ReportLab. "
            "Supports headers, styled tables, bullet lists, and automatic page numbering."
        )

    @property
    def permission(self) -> ToolPermission:
        return ToolPermission.READ_ONLY

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Title of the document"},
                "content": {"type": "string", "description": "Body text or formatted paragraphs"},
                "sections": {
                    "type": "array",
                    "description": "Optional structured list of sections with heading, text, bullets, table",
                    "items": {"type": "object"},
                },
                "table_data": {
                    "type": "object",
                    "description": "Optional table data with 'headers' and 'rows'",
                    "properties": {
                        "headers": {"type": "array", "items": {"type": "string"}},
                        "rows": {"type": "array", "items": {"type": "array", "items": {"type": "string"}}},
                    },
                },
                "filename": {"type": "string", "description": "Optional custom filename ending with .pdf"},
                "author": {"type": "string", "description": "Author name", "default": "A.O.R.I.S."},
            },
            "required": ["title"],
        }

    def execute(
        self,
        title: str,
        content: Optional[str] = None,
        sections: Optional[List[Dict[str, Any]]] = None,
        table_data: Optional[Dict[str, Any]] = None,
        filename: Optional[str] = None,
        author: str = "A.O.R.I.S.",
        **kwargs: Any,
    ) -> ToolResult:
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.lib import colors
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
            from reportlab.pdfgen import canvas
        except ImportError as exc:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=f"ReportLab is not installed: {exc}",
            )

        try:
            # 1. Sanitize filename
            if filename:
                clean_name = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", Path(filename).name)
                if not clean_name.lower().endswith(".pdf"):
                    clean_name += ".pdf"
            else:
                timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                slug = re.sub(r"[^a-zA-Z0-9_]", "_", title.lower()[:30]).strip("_")
                clean_name = f"{slug or 'report'}_{timestamp}.pdf"

            out_dir = self.output_dir
            out_dir.mkdir(parents=True, exist_ok=True)
            output_path = (out_dir / clean_name).resolve()

            # 2. Numbered Canvas for page numbers and footer
            class _NumberedCanvas(canvas.Canvas):
                def __init__(self, *args: Any, **kwargs: Any) -> None:
                    super().__init__(*args, **kwargs)
                    self._saved_page_states: List[Dict[str, Any]] = []

                def showPage(self) -> None:
                    self._saved_page_states.append(dict(self.__dict__))
                    self._startPage()

                def save(self) -> None:
                    num_pages = len(self._saved_page_states)
                    for state in self._saved_page_states:
                        self.__dict__.update(state)
                        self.draw_page_decorations(num_pages)
                        super().showPage()
                    super().save()

                def draw_page_decorations(self, total_pages: int) -> None:
                    self.saveState()
                    self.setFont("Helvetica", 8)
                    self.setFillColor(colors.HexColor("#64748B"))
                    # Footer rule
                    self.setStrokeColor(colors.HexColor("#E2E8F0"))
                    self.setLineWidth(0.75)
                    self.line(54, 42, letter[0] - 54, 42)
                    # Footer labels
                    self.drawString(54, 28, f"{author} — Multi-Model AI Orchestration System")
                    self.drawRightString(letter[0] - 54, 28, f"Page {self._pageNumber} of {total_pages}")
                    self.restoreState()

            # 3. Document setup & Styles
            doc = SimpleDocTemplate(
                str(output_path),
                pagesize=letter,
                leftMargin=54,
                rightMargin=54,
                topMargin=54,
                bottomMargin=54,
            )
            styles = getSampleStyleSheet()

            # Custom typography styles
            title_style = ParagraphStyle(
                "DocTitle",
                parent=styles["Normal"],
                fontName="Helvetica-Bold",
                fontSize=20,
                leading=24,
                textColor=colors.HexColor("#0F172A"),
                spaceAfter=6,
            )
            meta_style = ParagraphStyle(
                "DocMeta",
                parent=styles["Normal"],
                fontName="Helvetica",
                fontSize=9,
                leading=12,
                textColor=colors.HexColor("#64748B"),
                spaceAfter=14,
            )
            heading_style = ParagraphStyle(
                "DocHeading",
                parent=styles["Heading2"],
                fontName="Helvetica-Bold",
                fontSize=13,
                leading=16,
                textColor=colors.HexColor("#1E293B"),
                spaceBefore=14,
                spaceAfter=6,
            )
            body_style = ParagraphStyle(
                "DocBody",
                parent=styles["BodyText"],
                fontName="Helvetica",
                fontSize=10,
                leading=14,
                textColor=colors.HexColor("#334155"),
                spaceAfter=8,
            )
            bullet_style = ParagraphStyle(
                "DocBullet",
                parent=styles["Normal"],
                fontName="Helvetica",
                fontSize=10,
                leading=14,
                textColor=colors.HexColor("#334155"),
                leftIndent=15,
                spaceAfter=4,
            )
            table_cell_style = ParagraphStyle(
                "TableCell",
                parent=styles["Normal"],
                fontName="Helvetica",
                fontSize=9,
                leading=12,
                textColor=colors.HexColor("#1E293B"),
            )
            table_header_style = ParagraphStyle(
                "TableHeader",
                parent=styles["Normal"],
                fontName="Helvetica-Bold",
                fontSize=9,
                leading=12,
                textColor=colors.white,
            )

            story: List[Any] = []

            # 4. Title & Metadata
            story.append(Paragraph(title, title_style))
            gen_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            meta_text = f"Author: <b>{author}</b> &nbsp;|&nbsp; Generated: {gen_time} &nbsp;|&nbsp; System: A.O.R.I.S."
            story.append(Paragraph(meta_text, meta_style))
            story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#CBD5E1"), spaceAfter=14))

            # 5. Process free-form content
            if content:
                paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
                for p_text in paragraphs:
                    if p_text.startswith("### "):
                        story.append(Paragraph(p_text[4:].strip(), heading_style))
                    elif p_text.startswith("## "):
                        story.append(Paragraph(p_text[3:].strip(), heading_style))
                    elif p_text.startswith("# "):
                        story.append(Paragraph(p_text[2:].strip(), heading_style))
                    elif p_text.startswith("- ") or p_text.startswith("* "):
                        for item in p_text.split("\n"):
                            cleaned = item.strip().lstrip("-* ").strip()
                            if cleaned:
                                story.append(Paragraph(f"&bull; {cleaned}", bullet_style))
                    else:
                        story.append(Paragraph(p_text.replace("\n", "<br/>"), body_style))

            # 6. Process structured sections
            if sections:
                for sec in sections:
                    if not isinstance(sec, dict):
                        continue
                    if sec.get("heading"):
                        story.append(Paragraph(sec["heading"], heading_style))
                    if sec.get("text"):
                        story.append(Paragraph(sec["text"].replace("\n", "<br/>"), body_style))
                    if sec.get("bullets"):
                        for b in sec["bullets"]:
                            story.append(Paragraph(f"&bull; {str(b)}", bullet_style))
                    if sec.get("table") and isinstance(sec["table"], dict):
                        t_headers = sec["table"].get("headers", [])
                        t_rows = sec["table"].get("rows", [])
                        if t_headers or t_rows:
                            tbl_data: List[List[Any]] = []
                            if t_headers:
                                tbl_data.append([Paragraph(str(h), table_header_style) for h in t_headers])
                            for r in t_rows:
                                tbl_data.append([Paragraph(str(c), table_cell_style) for c in r])
                            t = Table(tbl_data, hAlign="LEFT")
                            t.setStyle(TableStyle([
                                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E293B")),
                                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                                ("TOPPADDING", (0, 0), (-1, -1), 5),
                                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
                                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                            ]))
                            story.append(Spacer(1, 6))
                            story.append(t)
                            story.append(Spacer(1, 10))

            # 7. Process standalone table_data
            if table_data and isinstance(table_data, dict):
                t_headers = table_data.get("headers", [])
                t_rows = table_data.get("rows", [])
                if t_headers or t_rows:
                    tbl_data: List[List[Any]] = []
                    if t_headers:
                        tbl_data.append([Paragraph(str(h), table_header_style) for h in t_headers])
                    for r in t_rows:
                        tbl_data.append([Paragraph(str(c), table_cell_style) for c in r])
                    t = Table(tbl_data, hAlign="LEFT")
                    t.setStyle(TableStyle([
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E293B")),
                        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                        ("TOPPADDING", (0, 0), (-1, -1), 5),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                    ]))
                    story.append(Spacer(1, 6))
                    story.append(t)
                    story.append(Spacer(1, 10))

            # Build PDF
            doc.build(story, canvasmaker=_NumberedCanvas)

            # 8. Post-generation verification
            if not output_path.exists():
                return ToolResult(tool_name=self.name, success=False, error="PDF file was not created on disk.")

            file_size = output_path.stat().st_size
            if file_size == 0:
                return ToolResult(tool_name=self.name, success=False, error="Generated PDF file is empty (0 bytes).")

            with open(output_path, "rb") as f:
                header = f.read(10)
                f.seek(max(0, file_size - 1024))
                tail = f.read(1024)

            if not header.startswith(b"%PDF-"):
                return ToolResult(
                    tool_name=self.name,
                    success=False,
                    error=f"Invalid PDF header: {header[:10]!r}",
                )

            is_valid_eof = b"%%EOF" in tail

            download_url = f"/api/downloads/{clean_name}"
            return ToolResult(
                tool_name=self.name,
                success=True,
                data={
                    "file_name": clean_name,
                    "file_path": str(output_path),
                    "bytes_written": file_size,
                    "download_url": download_url,
                    "title": title,
                    "author": author,
                    "verified": True,
                    "valid_header": True,
                    "valid_eof": is_valid_eof,
                },
            )
        except Exception as e:
            return ToolResult(tool_name=self.name, success=False, error=f"PDF generation failed: {e}")


