from os import environ
from typing import Optional, Any, Dict, Iterable

from cohere import Client as CohereClient
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class LLM:
    _client: Any
    system: str
    data: str

    def __init__(
            self,
            api_key: Optional[str] = None,
            system: Optional[str] = None,
            data: Optional[str] = None,
    ) -> None:
        if system is None:
            system = "You are a helpful assistant."
        self.system = system
        self.data = data
        self._client = CohereClient(api_key=environ["COHERE_API_KEY"] if api_key is None else api_key)

    def respond(
            self,
            query: str,
            **kwargs
    ) -> str:
        """Responds to the user's query."""
        system_prompt = build_system_prompt(
            system=self.system,
            sections={"data": self.data} if self.data is not None else {}
        )
        response = self._client.chat(
            model="command-r",
            preamble=system_prompt,
            chat_history=[],
            message=query
        )
        return response.text


def build_system_prompt(
        system: Optional[str] = "",
        sections: Optional[Dict[str, str]] = None
) -> str:
    """
    Builds the system prompt for a given LLM.
    """
    if sections is None:
        sections = {}
    prompt = system
    for key, section in sections.items():
        prompt += f"\n\n### {key.upper().strip()}\n{section}"
    return prompt


def split_corpus(
        corpus: Iterable[Document],
        chunk_size: int = 300,
        chunk_overlap: int = 20
):
    """Builds the retriever associated with the split provided corpus."""
    data = (
        RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len)
        .split_documents(corpus)
    )
    data = [d for d in data]
    return data
