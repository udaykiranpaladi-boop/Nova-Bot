"""
config.py - one place for every setting NovaBot uses.

Why this file exists
--------------------
Before, the embedding model name and the vector-store path were copy-pasted
into both ``ingestion.py`` and ``chatbot.py``. If one copy changed and the
other did not, the bot would silently stop finding answers. Now both files
import the same values from here.

All paths are built from this file's own location (``Path(__file__)``), so
the project works no matter which folder you launch it from.

Two settings can be overridden with environment variables (or in your
``.env`` file) without touching any code:

* ``NOVABOT_LLM_MODEL``  - which Groq-hosted model writes the answers.
* ``NOVABOT_TOP_K``      - how many document chunks to hand to the model.
"""

import os
from pathlib import Path

# --- Folders ---------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

# Where your source files (PDF / DOCX / TXT) live. Drop new files in here.
DOCUMENTS_DIR = BASE_DIR / "documents"

# Where the searchable index (ChromaDB) is saved after ingestion.
VECTOR_DB_PATH = BASE_DIR / "vector_store"

# --- Models ----------------------------------------------------------------
# Turns text into numbers ("embeddings") so we can search by meaning.
# Runs locally on your machine - no API key needed. It is downloaded
# automatically the first time you run the project (roughly 90 MB).
# IMPORTANT: ingestion and chatting MUST use the same embedding model.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# The language model that reads the retrieved text and writes the answer.
# It is served by Groq, so a free GROQ_API_KEY is required.
LLM_MODEL = os.getenv("NOVABOT_LLM_MODEL", "openai/gpt-oss-120b")

# Low temperature = more factual, less "creative". Good for a support bot.
LLM_TEMPERATURE = 0.3

# --- Retrieval -------------------------------------------------------------
# How many of the best-matching chunks are given to the model per question.
TOP_K = int(os.getenv("NOVABOT_TOP_K", "4"))

# --- Ingestion: semantic chunking -------------------------------------------
# A new chunk starts where the topic changes "sharply". 95 means: split at
# the top 5% biggest topic changes. Lower number = more, smaller chunks.
CHUNK_BREAKPOINT_TYPE = "percentile"
CHUNK_BREAKPOINT_AMOUNT = 95
