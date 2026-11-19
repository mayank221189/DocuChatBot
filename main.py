from io import BytesIO
from typing import cast

import streamlit as st
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

OLLAMA_BASE_URL = "http://localhost:11434"
CHAT_MODEL = "llama3.2:3b"
EMBEDDING_MODEL = "nomic-embed-text"


def load_pdf_documents(pdf_content: bytes, source_name: str) -> list[Document]:
    reader = PdfReader(BytesIO(pdf_content))
    total_pages = len(reader.pages)

    return [
        Document(
            page_content=(page.extract_text() or "").strip(),
            metadata={
                "source": source_name,
                "total_pages": total_pages,
                "page": page_number,
                "page_label": reader.page_labels[page_number],
            },
        )
        for page_number, page in enumerate(reader.pages)
    ]


def build_vector_store(
    pdf_content: bytes,
    source_name: str,
) -> InMemoryVectorStore:
    documents = load_pdf_documents(pdf_content, source_name)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )
    return InMemoryVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
    )


def answer_question(
    query: str,
    vector_store: InMemoryVectorStore,
    chat_model: ChatOllama,
) -> str:
    documents = vector_store.similarity_search(query, k=2)
    context = "\n\n".join(document.page_content for document in documents)
    prompt = (
        "Answer the following question based on the context provided.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )
    return chat_model.invoke(prompt).text


def on_document_upload_change() -> None:
    st.session_state.document_uploaded = False
    st.session_state.pop("vector_store", None)
    st.session_state.messages = []


def main() -> None:
    chat_model = ChatOllama(
        model=CHAT_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
    )

    st.title("Document Chatbot")
    st.subheader("Ask questions about the document")

    if "document_uploaded" not in st.session_state:
        st.session_state.document_uploaded = False
    if "messages" not in st.session_state:
        st.session_state.messages = []

    uploaded_file = st.file_uploader(
        "Upload a PDF document",
        type=["pdf"],
        key="document_uploader",
        on_change=on_document_upload_change,
    )

    if uploaded_file is not None and not st.session_state.document_uploaded:
        with st.spinner("Adding document..."):
            vector_store = build_vector_store(
                uploaded_file.getvalue(),
                uploaded_file.name,
            )

        st.session_state.vector_store = vector_store
        st.session_state.document_uploaded = True
        st.success(f"{uploaded_file.name} uploaded and indexed successfully.")

    if st.session_state.document_uploaded:
        messages = cast(
            list[dict[str, str]],
            st.session_state.get("messages", []),
        )
        vector_store = cast(
            InMemoryVectorStore,
            st.session_state.vector_store,
        )

        for message in messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        query = st.chat_input("Enter your question:")
        if query:
            messages.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)

            with st.spinner("Generating response..."):
                answer = answer_question(query, vector_store, chat_model)

            messages.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.markdown(answer)


main()
