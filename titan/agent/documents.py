"""Lectura de documentos: texto, Markdown, CSV, JSON, PDF, Word, Excel y PowerPoint.

Los formatos binarios usan dependencias opcionales (``pip install -e ".[docs]"``):
pypdf, python-docx, openpyxl y python-pptx. Si falta una, se informa con un
error claro en lugar de fallar en silencio.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

TEXT_EXT = {".txt", ".md", ".py", ".json", ".yaml", ".yml", ".csv", ".tsv", ".html", ".htm", ".xml",
            ".js", ".ts", ".c", ".cpp", ".h", ".java", ".rs", ".go", ".sql", ".ini", ".toml", ".log"}
SUPPORTED = TEXT_EXT | {".pdf", ".docx", ".xlsx", ".pptx"}


def _need(mod: str, pkg: str):
    try:
        return __import__(mod)
    except ImportError as e:
        raise RuntimeError(f"Para leer este formato instale {pkg} (pip install -e \".[docs]\")") from e


def read_document(path: str | Path, max_chars: int | None = None) -> str:
    p = Path(path)
    ext = p.suffix.lower()
    if ext in TEXT_EXT:
        raw = p.read_bytes()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("latin-1")
    elif ext == ".pdf":
        pypdf = _need("pypdf", "pypdf")
        reader = pypdf.PdfReader(str(p))
        text = "\n\n".join(f"[página {i + 1}]\n{page.extract_text() or ''}" for i, page in enumerate(reader.pages))
    elif ext == ".docx":
        docx = _need("docx", "python-docx")
        d = docx.Document(str(p))
        parts = [para.text for para in d.paragraphs]
        for t in d.tables:
            for row in t.rows:
                parts.append(" | ".join(c.text for c in row.cells))
        text = "\n".join(parts)
    elif ext == ".xlsx":
        openpyxl = _need("openpyxl", "openpyxl")
        wb = openpyxl.load_workbook(str(p), read_only=True, data_only=True)
        parts = []
        for ws in wb.worksheets:
            parts.append(f"[hoja {ws.title}]")
            for row in ws.iter_rows(values_only=True):
                parts.append(",".join("" if v is None else str(v) for v in row))
        text = "\n".join(parts)
    elif ext == ".pptx":
        pptx = _need("pptx", "python-pptx")
        prs = pptx.Presentation(str(p))
        parts = []
        for i, slide in enumerate(prs.slides):
            parts.append(f"[diapositiva {i + 1}]")
            for shape in slide.shapes:
                if shape.has_text_frame:
                    parts.append(shape.text_frame.text)
        text = "\n".join(parts)
    else:
        raise ValueError(f"Formato no soportado: {ext}")
    if max_chars is not None and len(text) > max_chars:
        text = text[:max_chars]
    return text


def summarize_table(path: str | Path, max_rows: int = 100_000) -> dict:
    """Estadísticas básicas por columna de un CSV (análisis de datos sin dependencias)."""
    p = Path(path)
    text = read_document(p)
    sample = text[:4096]
    dialect = csv.Sniffer().sniff(sample) if sample.strip() else csv.excel
    rows = list(csv.reader(io.StringIO(text), dialect))
    if not rows:
        return {"rows": 0, "columns": []}
    header, data = rows[0], rows[1:max_rows + 1]
    cols = []
    for j, name in enumerate(header):
        vals = [r[j] for r in data if j < len(r) and r[j] != ""]
        nums = []
        for v in vals:
            for cand in (v, v.replace(",", "")):  # admite separador de miles "1,234.5"
                try:
                    nums.append(float(cand))
                    break
                except ValueError:
                    continue
        info = {"name": name, "non_empty": len(vals), "unique": len(set(vals))}
        if nums and len(nums) >= 0.8 * len(vals):
            info.update({"type": "numérica", "min": min(nums), "max": max(nums), "mean": sum(nums) / len(nums)})
        else:
            info.update({"type": "texto", "top": sorted(set(vals), key=vals.count, reverse=True)[:5]})
        cols.append(info)
    return {"rows": len(data), "columns": cols, "preview": json.dumps(rows[:4], ensure_ascii=False)}


def write_document(path: str | Path, content) -> dict:
    """Crea un documento según la extensión.

    * .docx  ``content``: texto; las líneas que empiezan por "# " son títulos.
    * .xlsx  ``content``: lista de filas (listas) o texto CSV.
    * .pptx  ``content``: texto con diapositivas separadas por "# Título" y viñetas "- ".
    * .md/.txt/.csv/.json  se escribe tal cual (JSON si no es texto).
    """
    p = Path(path)
    ext = p.suffix.lower()
    p.parent.mkdir(parents=True, exist_ok=True)
    if ext == ".docx":
        docx = _need("docx", "python-docx")
        d = docx.Document()
        for line in str(content).split("\n"):
            if line.startswith("# "):
                d.add_heading(line[2:].strip(), level=1)
            elif line.startswith("## "):
                d.add_heading(line[3:].strip(), level=2)
            elif line.startswith("- "):
                d.add_paragraph(line[2:].strip(), style="List Bullet")
            elif line.strip():
                d.add_paragraph(line)
        d.save(str(p))
    elif ext == ".xlsx":
        openpyxl = _need("openpyxl", "openpyxl")
        rows = content if isinstance(content, list) else list(csv.reader(io.StringIO(str(content))))
        wb = openpyxl.Workbook()
        ws = wb.active
        for r in rows:
            ws.append([_cell(v) for v in (r if isinstance(r, list) else [r])])
        wb.save(str(p))
    elif ext == ".pptx":
        pptx = _need("pptx", "python-pptx")
        prs = pptx.Presentation()
        slides = [s for s in str(content).split("\n# ") if s.strip()]
        for block in slides:
            lines = block.lstrip("# ").split("\n")
            slide = prs.slides.add_slide(prs.slide_layouts[1])
            slide.shapes.title.text = lines[0].strip()
            body = slide.placeholders[1].text_frame
            bullets = [ln[2:].strip() if ln.startswith("- ") else ln.strip() for ln in lines[1:] if ln.strip()]
            body.text = bullets[0] if bullets else ""
            for b in bullets[1:]:
                body.add_paragraph().text = b
        prs.save(str(p))
    elif ext in TEXT_EXT:
        p.write_text(content if isinstance(content, str) else json.dumps(content, ensure_ascii=False, indent=2),
                     encoding="utf-8")
    else:
        raise ValueError(f"Formato no soportado para escribir: {ext}")
    return {"path": str(p), "bytes": p.stat().st_size}


def _cell(v):
    if isinstance(v, str):
        for cast in (int, float):
            try:
                return cast(v)
            except ValueError:
                pass
    return v
