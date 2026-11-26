"""
ingestion.py - builds NovaBot's searchable "knowledge index".

Run this ONCE, and again whenever you add or change files in ``documents/``:

    python ingestion.py

What happens, step by step
--------------------------
1. LOAD   - read every file in ``documents/`` (see ``parsing.py``).
2. CHUNK  - cut the text into small pieces. Instead of cutting every N
            characters, we use *semantic chunking*: the splitter measures how
            similar neighbouring sentences are and cuts where the topic
            changes. Each chunk therefore stays about one idea.
3. EMBED  - turn each chunk into a list of numbers (an "embedding") that
            captures its meaning.
4. STORE  - save the chunks + embeddings in a local ChromaDB folder
            (``vector_store/``). ``chatbot.py`` searches this folder later.
"""

from langchain_chroma import Chroma
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

from config import (
    CHUNK_BREAKPOINT_AMOUNT,
    CHUNK_BREAKPOINT_TYPE,
    DOCUMENTS_DIR,
    EMBEDDING_MODEL,
    VECTOR_DB_PATH,
)
from parsing import parse_folder


def chunk_documents(documents, embeddings):
    """Split documents into topic-sized chunks using semantic chunking.

    Args:
        documents: Documents returned by ``parse_folder``.
        embeddings: The embedding model used to measure sentence similarity.

    Returns:
        A list of smaller ``Document`` chunks (metadata such as the source
        file name is carried over to every chunk).
    """
    splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type=CHUNK_BREAKPOINT_TYPE,
        breakpoint_threshold_amount=CHUNK_BREAKPOINT_AMOUNT,
    )
    chunks = splitter.split_documents(documents)
    print(f"Total chunks created: {len(chunks)}")
    return chunks


def build_vector_store(chunks, embeddings):
    """Embed the chunks and save them to the ChromaDB folder.

    Any older index is cleared first. Otherwise, running ingestion twice
    would store every chunk twice and the bot would see duplicate context.

    Args:
        chunks: Chunks returned by ``chunk_documents``.
        embeddings: The embedding model (must match the one used in chatbot.py).

    Returns:
        The ``Chroma`` vector store that was just built.
    """
    if VECTOR_DB_PATH.exists():
        print(f"Clearing old index at: {VECTOR_DB_PATH}")
        Chroma(
            persist_directory=str(VECTOR_DB_PATH),
            embedding_function=embeddings,
        ).delete_collection()

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(VECTOR_DB_PATH),
    )
    print(f"Vector store saved to: {VECTOR_DB_PATH}")
    return vector_store


def main():
    """Run the full pipeline: load -> chunk -> embed -> store."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    documents = parse_folder(DOCUMENTS_DIR)
    if not documents:
        print(f"No supported documents found in {DOCUMENTS_DIR}. Nothing to do.")
        return
    chunks = chunk_documents(documents, embeddings)
    build_vector_store(chunks, embeddings)


if __name__ == "__main__":
    main()
