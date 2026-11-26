"""
app.py - the web interface for NovaBot (built with Streamlit).

Start it with:

    streamlit run app.py

Streamlit re-runs this whole file from top to bottom every time the user
clicks or types something. That is why the heavy parts (loading models) are
cached, and why the chat messages are kept in ``st.session_state``.
"""

import uuid

import streamlit as st
from langchain_core.runnables import RunnableConfig

from chatbot import build_chain

st.set_page_config(page_title="NovaBot", page_icon="💬")
st.title("💬 NovaBot — NovaCorp Assistant")
st.caption("Ask questions about NovaCorp. Answers come only from the company documents.")


@st.cache_resource
def get_chain():
    """Build the chatbot once and reuse it on every re-run.

    Without the cache, the embedding model and the vector store would be
    reloaded after every single message, which is very slow.
    """
    return build_chain()


try:
    chain = get_chain()
except RuntimeError as err:
    # Setup problems (missing API key / missing index) get a friendly message
    # on screen instead of a scary traceback.
    st.error(str(err))
    st.stop()

# Streamlit's own memory (per browser tab). It is only used to redraw the
# chat bubbles; the model's conversational memory lives in chatbot.py.
if "display_messages" not in st.session_state:
    st.session_state.display_messages = []

# A unique id per browser tab, so different users never share a conversation.
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# --- Sidebar ---------------------------------------------------------------
with st.sidebar:
    st.header("About")
    st.write(
        "NovaBot uses **RAG**: it looks up the most relevant parts of the "
        "NovaCorp handbook first, then answers using only those parts."
    )
    st.subheader("Try asking")
    st.markdown(
        "- Who is the CEO of NovaCorp?\n"
        "- How many PTO days do employees get?\n"
        "- What is the hybrid work policy?\n"
        "- What is the PRO-Engine?\n"
        "- What is the capital of France? *(it should politely refuse)*"
    )
    if st.button("New conversation"):
        # A fresh session id = a fresh, empty memory for the model.
        st.session_state.display_messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

# --- Chat area -------------------------------------------------------------
# Redraw earlier messages.
for msg in st.session_state.display_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask NovaBot something about NovaCorp...")

if question:
    st.session_state.display_messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            config: RunnableConfig = {
                "configurable": {"session_id": st.session_state.session_id}
            }
            try:
                answer = chain.invoke({"question": question}, config=config)
            except Exception as err:  # network down, rate limit, bad key, ...
                answer = f"Sorry, something went wrong while contacting the model: {err}"
        st.markdown(answer)

    st.session_state.display_messages.append({"role": "assistant", "content": answer})
