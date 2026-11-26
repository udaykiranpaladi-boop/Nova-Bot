"""
parsing.py - reads every supported file inside a folder.

This is step 1 of ingestion: "get all the raw text into memory".
It looks at each file's extension, picks the matching loader from
``loader.LOADER_MAP``, and collects everything into one list.

Run it on its own to check that your documents load correctly:

    python parsing.py
"""

from pathlib import Path

from config import DOCUMENTS_DIR
from loader import LOADER_MAP


def parse_folder(folder_path: Path = DOCUMENTS_DIR):
    """Load every supported file (.pdf, .docx, .txt) in ``folder_path``.

    Args:
        folder_path: Folder to scan. Defaults to the project's ``documents/``
            folder. Sub-folders are ignored.

    Returns:
        A list of LangChain ``Document`` objects (a PDF gives one per page).
        Files with unsupported extensions are skipped with a message.

    Raises:
        FileNotFoundError: If the folder does not exist.
    """
    folder_path = Path(folder_path)
    if not folder_path.is_dir():
        raise FileNotFoundError(f"Documents folder not found: {folder_path}")

    all_documents = []
    # sorted() makes the order the same on every run.
    for file_path in sorted(folder_path.iterdir()):
        if not file_path.is_file():
            continue

        loader_func = LOADER_MAP.get(file_path.suffix.lower())
        if loader_func is None:
            print(f"Skipping unsupported file: {file_path.name}")
            continue

        print(f"Loading: {file_path.name}")
        all_documents.extend(loader_func(str(file_path)))

    print(f"Total documents loaded: {len(all_documents)}")
    return all_documents


if __name__ == "__main__":
    docs = parse_folder()
    print(docs[0] if docs else "No documents found.")
