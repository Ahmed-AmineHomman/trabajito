from typing import Iterable

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from unstructured.partition.auto import partition


def load_corpus(filepath: str) -> Iterable[Document]:
    """Loads a document from a given file path."""
    chunks = partition(filename=filepath)
    return [Document(page_content=chunk.text, metadata={"source": filepath}) for chunk in chunks]


def split_corpus(
        corpus: Iterable[Document],
        chunk_size: int = 300,
        chunk_overlap: int = 20
):
    """Builds the retriever associated with the split provided corpus."""
    data = (
        RecursiveCharacterTextSplitter(
            separators=["\n"],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        .split_documents(corpus)
    )
    data = [d for d in data]
    return data
