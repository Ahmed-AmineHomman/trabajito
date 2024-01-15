from os import environ
from typing import Optional

from openai import OpenAI


class BaseLLM:
    _messages: list = []
    context: str = ""
    data: str = ""

    def __init__(
            self,
            context: Optional[str] = "",
            data: Optional[str] = ""
    ) -> None:
        self.context = context if context is not None else ""
        self.data = data if data is not None else ""

    def respond(
            self,
            query: str
    ) -> str:
        raise NotImplementedError

    def _build_context(self) -> str:
        return f"{self.context}\n COURS:\n{self.data}"


class OpenAILLM(BaseLLM):
    _client: OpenAI = None

    def __init__(
            self,
            context: Optional[str] = "",
            data: Optional[str] = "",
            api_key: Optional[str] = None,
    ):
        super().__init__(context=context, data=data)
        self.authenticate(api_key=api_key)

    def authenticate(
            self,
            api_key: Optional[str] = None
    ) -> None:
        self._client = OpenAI(api_key=environ["OPENAI_API_KEY"] if api_key is None else api_key)

    def respond(
            self,
            query: str,
    ) -> str:
        context = self._build_context()
        self._messages.append({"role": "user", "content": query})
        response = self._client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": context}] + self._messages
        ).choices[0].message.content
        self._messages.append({"role": "assistant", "content": response})
        return response
