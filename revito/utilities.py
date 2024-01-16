from langchain_community.document_loaders import UnstructuredPDFLoader as PDFLoader, \
    UnstructuredWordDocumentLoader as WordLoader, UnstructuredHTMLLoader as HTMLLoader
from langchain_community.vectorstores import FAISS
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings


def build_retriever(
        filepath: str,
):
    # load data from document
    extension = filepath.split(".")[-1].lower()
    try:
        if extension == "pdf":
            data = PDFLoader(filepath).load()
        elif extension in ["doc", "docx", "odt"]:
            data = WordLoader(filepath).load()
        elif extension == "html":
            data = HTMLLoader(filepath).load()
        else:
            raise ValueError("Unsupported file type")
    except Exception as error:
        raise Exception(f"### ERROR (loading): {error}")

    # split text
    try:
        data = SemanticChunker(OpenAIEmbeddings()).split_documents(data)
        data = [d for d in data]
    except Exception as error:
        raise Exception(f"### ERROR (splitting): {error}")

    # compute vector store
    try:
        retriever = FAISS.from_documents(data, OpenAIEmbeddings()).as_retriever()
    except Exception as error:
        raise Exception(f"### ERROR (embedding): {error}")

    return retriever
