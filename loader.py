"""
loader.py - reads a single file and turns it into LangChain ``Document`` objects.

A ``Document`` is just a small container with two things:
  * ``page_content`` - the text
  * ``metadata``     - where the text came from (file name, page number, ...)

Each supported file type has its own loader function. ``LOADER_MAP`` links a
file extension (like ".pdf") to the right function, so ``parsing.py`` can pick
the correct one automatically.

To support a new file type, write a ``load_xxx`` function here and add one
line to ``LOADER_MAP``. Nothing else needs to change.
"""

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader


def load_pdf(file_path: str):
    """Read a PDF. Returns one Document per page (page number kept in metadata)."""
    return PyPDFLoader(file_path).load()


def load_docx(file_path: str):
    """Read a Word (.docx) file. Returns the whole file as one Document."""
    return Docx2txtLoader(file_path).load()


def load_txt(file_path: str):
    """Read a plain text file (UTF-8). Returns the whole file as one Document."""
    return TextLoader(file_path, encoding="utf-8").load()


# File extension -> function that knows how to read it.
LOADER_MAP = {
    ".pdf": load_pdf,
    ".docx": load_docx,
    ".txt": load_txt,
}
