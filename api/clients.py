import os
from typing import List, Optional, Dict

from cohere import Client as CohereClient


class ChatExchange:
    """Represents a conversation between a user and an AI assistant."""
    query: str
    response: str

    def __init__(self, query: str, response: str):
        self.query = query
        self.response = response


class BaseClient:
    conversation: List[ChatExchange]
    system_prompt: str

    def respond(
            self,
            query: str,
            system_prompt: Optional[str] = "You are a helpful assistant.",
            conversation: Optional[List[ChatExchange]] = None,
            data: Dict[str, str] = None,
            temperature: Optional[float] = 1.0,
    ) -> str:
        """
        Responds to the user's query according to the provided system prompt and past conversation.

        Parameters
        ----------
        query: str
            The user's query.
        system_prompt: str, optional
            The system prompt.
        conversation: List[ChatExchange], optional
            The past conversation.
        data: Dict[str, str], optional
            Additional data that the LLM can quote when answering the query.
        temperature: float, optional, default=1.0
            The temperature of the LLM when sampling its response.
            Higher values lead to more "creative" responses when lower values lead to more deterministic responses.
        """
        raise NotImplementedError()


class Cohere(BaseClient):
    """
    Implements the Cohere API client.
    """
    _environ_key: str = "COHERE_API_KEY"
    api_key: str
    client: CohereClient

    def __init__(self, api_key: Optional[str] = None):
        self.client = CohereClient(api_key=api_key if api_key else os.getenv(self._environ_key))

    def respond(
            self,
            query: str,
            system_prompt: Optional[str] = "You are a helpful assistant.",
            conversation: Optional[List[ChatExchange]] = None,
            data: Dict[str, str] = None,
            temperature: Optional[float] = 1.0,
            model: str = "command-r"
    ) -> str:
        """See base class for details."""
        if conversation is None:
            conversation = []
        if data is None:
            data = {}

        # cast conversation to chat history
        chat_history = []
        for exchange in conversation:
            chat_history += [
                {"role": "USER", "message": exchange.query},
                {"role": "CHATBOT", "message": exchange.response},
            ]

        # build corpus
        documents = []
        for title, content in data.items():
            documents.append({"title": title, "text": content})

        # request response
        response = self.client.chat(
            model=model,
            preamble=system_prompt,
            chat_history=chat_history,
            message=query,
            documents=documents,
            temperature=temperature,
        )

        return response.text
