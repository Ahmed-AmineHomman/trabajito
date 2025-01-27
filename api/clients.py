import os
from typing import List, Optional, Dict

import torch
from cohere import Client as CohereClient
from transformers import pipeline


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


class TransformersClient(BaseClient):
    pipe: pipeline
    device: str
    model_id: str = None

    def __init__(
            self,
            model_id: Optional[str] = None,
    ):
        """
        Initializes the chatbot with a specific model from Hugging Face.
        Args:
            model_id (str): Model identifier on Hugging Face Hub (e.g., 'gpt2').
            device (str): Device to load the model onto ('cuda' or 'cpu'). Auto-detected if not specified.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if model_id:
            self.load(model_id)

    def load(self, model_id: str):
        """
        Loads the model from the provided deposit identifier in the Hugging Face Hub.
        """
        load_model = False
        if self.model_id is None:
            self.model_id = model_id
            load_model = True
        if self.model_id != model_id:
            self.model_id = model_id
            load_model = True
        if load_model:
            self.pipe = pipeline("text-generation", model=model_id, device=self.device)

    def respond(
            self,
            query: str,
            system_prompt: Optional[str] = "You are a helpful assistant.",
            conversation: Optional[List[ChatExchange]] = None,
            data: Dict[str, str] = None,
            temperature: Optional[float] = 1.0,
            max_size: int = 512,
            model: Optional[str] = None
    ) -> str:

        """
        Responds to the user's query according to the provided system prompt and past conversation.

        Parameters
        ----------
        query : str
            The user's query.
        system_prompt : str, optional
            The system prompt.
        conversation : List[ChatExchange], optional
            The past conversation.
        data : Dict[str, str], optional
            Additional data that the LLM can quote when answering the query.
        temperature : float, optional, default=1.0
            The temperature of the LLM when sampling its response.
            Higher values lead to more "creative" responses when lower values lead to more deterministic responses.
        max_size: int, optional
            The maximum number of tokens to generate.
        model : str, optional
            The model identifier to use for generating the response.

        Returns
        -------
        str
            The generated response from the model.
        """
        if model:
            self.load(model)
        if conversation is None:
            conversation = []
        if data is None:
            data = {}

        # build system prompt
        system_prompt = f"{system_prompt}\n\n"
        for title, content in data.items():  # add data to the system prompt
            system_prompt += f"\n\n{title}\n{content}"

        # cast conversation to chat history
        messages = [{"role": "system", "content": system_prompt}]
        for exchange in conversation:
            messages += [
                {"role": "user", "message": exchange.query},
                {"role": "assistant", "message": exchange.response}
            ]
        messages += [{"role": "user", "content": query}]

        # run inference
        response = self.pipe(
            messages,
            temperature=temperature,
            max_new_tokens=max_size,
            num_return_sequences=1
        )

        return response[0].get("generated_text")[-1].get("content")
