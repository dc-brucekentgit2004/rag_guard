"""Read uploaded documents (PDF, TXT, MD, DOCX) into plain text for the guard.

The text is cleaned so that PDF line breaks do not cut sentences in half:
page footers like "Page 3 of 7", signature lines (____) and checkbox symbols are removed,
and all whitespace is collapsed. The guard's NLI module then splits the text into
overlapping 250-word windows by itself, so one long string per document is fine.
"""
import io
import re


def _clean(text: str) -> str:
    lines = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if re.search(r"\bpage\s+\d+\s+(of\s+\d+)?\s*$", s, re.I):   # "... Page 3 of 7"
            continue
        if re.fullmatch(r"[_\-\s.:]*", s):                           # signature / rule lines
            continue
        s = re.sub(r"^[•□☐☑☒]\s*", "", s)   # bullets / checkboxes
        lines.append(s)
    text = " ".join(lines)
    text = re.sub(r"_{3,}", "____", text)
    return re.sub(r"\s+", " ", text).strip()


def read_document(name: str, data: bytes) -> str:
    """Return cleaned text from an uploaded file's name and bytes."""
    ext = name.lower().rsplit(".", 1)[-1]
    if ext == "pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            raise RuntimeError("PDF support needs one extra package. Run:  pip install pypdf")
        reader = PdfReader(io.BytesIO(data))
        pages = []
        for page in reader.pages:
            try:  # layout mode keeps words intact in PDFs with unusual spacing
                pages.append(page.extract_text(extraction_mode="layout") or "")
            except TypeError:  # older pypdf
                pages.append(page.extract_text() or "")
        raw = "\n".join(pages)
        if not raw.strip():
            raise RuntimeError(f"No text found in {name}. It may be a scanned image; use a text PDF.")
    elif ext == "docx":
        try:
            import docx
        except ImportError:
            raise RuntimeError("Word support needs one extra package. Run:  pip install python-docx")
        raw = "\n".join(p.text for p in docx.Document(io.BytesIO(data)).paragraphs)
    elif ext in ("txt", "md"):
        raw = data.decode("utf-8", errors="ignore")
    else:
        raise RuntimeError(f"Unsupported file type: .{ext} (use PDF, TXT, MD or DOCX)")
    return _clean(raw)
