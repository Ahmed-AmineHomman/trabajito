from os import environ
from typing import Optional, Any, List, Dict, Tuple

from cohere import Client as CohereClient
from openai import OpenAI

from .prompts import build_system_prompt
from copy import deepcopy


class BaseLLM:
    _messages: List[Dict[str, str]]
    _client: Any
    _api_entry: str
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
        self._messages = []
        self._authenticate(api_key=environ[self._api_entry] if api_key is None else api_key)

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
        conversation = self._build_history(query)
        response = self._respond(
            system=system_prompt,
            conversation=conversation,
            query=query,
            **kwargs
        )
        self._messages.append({"role": "user", "content": query})
        self._messages.append({"role": "assistant", "content": response})
        return response

    def reset_messages(self):
        self._messages = []

    def get_conversation(self) -> List[Tuple[str]]:
        n = int(len(self._messages) / 2)
        return [
            (self._messages[2 * i].get("content"), self._messages[2 * i + 1].get("content"))
            for i in range(n)
        ]

    def _authenticate(
            self,
            api_key: str,
    ) -> None:
        raise NotImplementedError()

    def _build_history(
            self,
            query: str,
    ) -> Any:
        raise NotImplementedError()

    def _respond(
            self,
            system: str,
            conversation: Any,
            query: str,
            **kwargs,
    ) -> str:
        raise NotImplementedError()


class OpenAILLM(BaseLLM):
    _client: OpenAI = None
    _api_entry = "OPENAI_API_KEY"

    def _authenticate(
            self,
            api_key: str,
    ) -> None:
        self._client = OpenAI(api_key=api_key)

    def _respond(
            self,
            system: str,
            conversation: List[Dict[str, str]],
            query: str,
            **kwargs
    ) -> str:
        messages = [
            {"role": "system", "content": system},
            *conversation,
            {"role": "user", "content": query}
        ]
        response = self._client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            **kwargs
        ).choices[0].message.content
        return response

    def _build_history(
            self,
            query: str,
    ) -> List[Dict[str, str]]:
        return [a for a in self._messages]


class CohereLLM(BaseLLM):
    _client: CohereClient
    _api_entry = "COHERE_API_KEY"
    _role_mapper = {
        "user": "USER",
        "assistant": "CHATBOT",
    }

    def _authenticate(
            self,
            api_key: str,
    ) -> None:
        self._client = CohereClient(api_key=api_key)

    def _respond(
            self,
            system: str,
            conversation: List[Dict[str, str]],
            query: str,
            **kwargs
    ) -> str:
        response = self._client.chat(
            model="command",
            preamble_override=system,
            chat_history=conversation,
            message=query
        )
        return response.text

    def _build_history(
            self,
            query: str,
    ) -> List[Dict[str, str]]:
        return [
            {"role": self._role_mapper.get(a.get("user")), "message": a.get("content")}
            for a in self._messages
        ]


class LLMFactory:
    def __call__(
            self,
            model: str,
            **kwargs
    ) -> BaseLLM:
        if model == "cohere":
            return CohereLLM(**kwargs)
        elif model == "openai":
            return OpenAILLM(**kwargs)
        else:
            raise ValueError(f"unrecognized model '{model}")
