import os
from pathlib import Path
from typing import cast

import streamlit as st
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import SecretStr
from pypdf import PdfReader


def load_pdf_documents(pdf_path: Path) -> list[Document]:
    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)
    source = str(pdf_path)

    return [
        Document(
            page_content=(page.extract_text() or "").strip(),
            metadata={
                "source": source,
                "total_pages": total_pages,
                "page": page_number,
                "page_label": reader.page_labels[page_number],
            },
        )
        for page_number, page in enumerate(reader.pages)
    ]


def build_vector_store(
    pdf_path: Path,
    api_key: str | None,
) -> InMemoryVectorStore:
    documents = load_pdf_documents(pdf_path)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2-preview",
        api_key=SecretStr(api_key) if api_key is not None else None,
    )
    return InMemoryVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
    )


def save_uploaded_document(filename: str, content: bytes) -> Path:
    documents_dir = Path(__file__).parent / "documents"
    documents_dir.mkdir(parents=True, exist_ok=True)

    uploaded_path = documents_dir / Path(filename).name
    for existing_file in documents_dir.iterdir():
        if existing_file.is_file():
            existing_file.unlink()

    uploaded_path.write_bytes(content)
    return uploaded_path


def answer_question(
    query: str,
    vector_store: InMemoryVectorStore,
    chat_model: ChatGoogleGenerativeAI,
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
    st.session_state.pop("uploaded_file_path", None)
    st.session_state.pop("vector_store", None)
    st.session_state.messages = []


def main() -> None:
    load_dotenv()
    api_key = os.getenv("GOOGLE_GEMINI_API_KEY")
    chat_model = ChatGoogleGenerativeAI(
        model="gemini-4-hydrogen",
        api_key=SecretStr(api_key) if api_key is not None else None,
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
            uploaded_path = save_uploaded_document(
                uploaded_file.name,
                uploaded_file.getvalue(),
            )
            vector_store = build_vector_store(uploaded_path, api_key)

        st.session_state.uploaded_file_path = str(uploaded_path)
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
