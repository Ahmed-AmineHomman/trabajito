import os
from typing import List, Optional, Dict

import numpy as np
import torch
from cohere import ClientV2, Document
from sentence_transformers import SentenceTransformer
from transformers import pipeline


class ChatExchange:
    """Represents a conversation between a user and an AI assistant."""
    query: str
    response: str

    def __init__(self, query: str, response: str):
        self.query = query
        self.response = response


class BaseClient:
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

    def embed(
            self,
            texts: List[str],
            model: Optional[str] = None
    ) -> np.ndarray:
        """
        Computes the embeddings of the provided texts.

        Parameters
        ----------
        texts: List[str]
            The texts to embed.
        model: str, optional
            The model identifier to use for generating the embeddings.

        Returns
        -------
        np.ndarray
            The embeddings of the texts.
        """
        raise NotImplementedError()


class CohereClient(BaseClient):
    """
    Implements the Cohere API client.
    """
    _environ_key: str = "COHERE_API_KEY"
    _client: ClientV2

    llm: str = "command-r"
    encoder: str = "embed-english-light-v3.0"

    def __init__(
            self,
            api_key: Optional[str] = None,
            llm: Optional[str] = None,
            encoder: Optional[str] = None
    ):
        self._client = ClientV2(api_key=api_key if api_key else os.getenv(self._environ_key))
        if llm:
            self.llm = llm
        if encoder:
            self.encoder = encoder

    def respond(
            self,
            query: str,
            system_prompt: Optional[str] = None,
            conversation: Optional[List[ChatExchange]] = None,
            data: Dict[str, str] = None,
            temperature: Optional[float] = 1.0,
            model: str = None
    ) -> str:
        """See base class for details."""
        if conversation is None:
            conversation = []
        if data is None:
            data = {}
        if model:
            self.llm = model

        # cast conversation to chat history
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        for exchange in conversation:
            messages += [
                {"role": "user", "content": exchange.query},
                {"role": "assistant", "content": exchange.response},
            ]
        messages.append({"role": "user", "content": query})

        # build documents
        documents = []
        for title, content in data.items():
            documents.append(Document(data={"title": title, "snipped": content}))

        # request response
        response = self._client.chat(
            model=self.llm,
            messages=messages,
            documents=documents,
            temperature=temperature,
        )

        return response.message.content[0].text

    def embed(
            self,
            texts: List[str],
            model: Optional[str] = "embed-english-light-v3.0",
            input_type: Optional[str] = "search_query"
    ) -> np.ndarray:
        """See base class for details."""
        response = self._client.embed(
            texts=texts,
            model=model,
            input_type=input_type,
            embedding_types=["float"],
        )
        return np.array(response.embeddings.float_)


class TransformersClient(BaseClient):
    """
    Wrappers around the ``transformers`` and ``diffusers`` APIs from Hugging Face.
    """
    pipe: pipeline
    device: str
    model_id: str = None

    def __init__(
            self,
            model: Optional[str] = None
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if model:
            self.pipe = pipeline("text-generation", model=model, device=self.device)

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
            self.pipe = pipeline("text-generation", model=model, device=self.device)
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


class SentenceTransformerClient(BaseClient):
    """
    Implements the SentenceTransformer API client.
    """
    model: SentenceTransformer = None
    device: str

    def __init__(
            self,
            use_cpu: Optional[bool] = False,
            model: Optional[str] = None
    ):
        self.device = "cpu" if (use_cpu or (not torch.cuda.is_available())) else "cuda"
        if model:
            self.model = SentenceTransformer(model)

    def embed(
            self,
            texts: List[str],
            model: Optional[str] = None,
            input_type: Optional[str] = "search_query"
    ) -> np.ndarray:
        """See base class for details."""
        if model:
            self.model = SentenceTransformer(model)
        embeddings = self.model.encode(
            [f"{input_type}: {t}" for t in texts],
            convert_to_tensor=True,
            device=self.device
        )
        if self.device == "cuda":
            embeddings = embeddings.cpu()
        return embeddings.numpy()
