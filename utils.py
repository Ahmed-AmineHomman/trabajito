from os import environ
from typing import Optional, Any, Dict, Iterable

from cohere import Client as CohereClient
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class LLM:
    _client: Any
    _messages: list
    _system: str
    data: str

    def __init__(
            self,
            model: Optional[str] = "command-r",
            system: Optional[str] = None,
            data: Optional[str] = None,
            api_key: Optional[str] = None,
    ) -> None:
        if system is None:
            system = "You are a helpful assistant."
        self.model = model
        self.data = data
        self._system = system
        self._messages = []
        self._client = CohereClient(api_key=environ["COHERE_API_KEY"] if api_key is None else api_key)

    def respond(
            self,
            query: str,
            **kwargs
    ) -> str:
        """Responds to the user's query."""
        system_prompt = build_system_prompt(
            system=self._system,
            sections={"data": self.data} if self.data is not None else {}
        )
        response = self._client.chat(
            model=self.model,
            preamble=system_prompt,
            chat_history=self._messages,
            message=query,
            **kwargs
        )
        self._messages.append({"role": "USER", "message": query})
        self._messages.append({"role": "CHATBOT", "message": response.text})
        return response.text

    def reset(self):
        self.data = ""
        self._messages = []


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
