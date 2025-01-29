import copy
from typing import List, Dict
from typing import Optional

import numpy as np
import torch
from transformers import DynamicCache
from unstructured.chunking.basic import chunk_elements
from unstructured.chunking.title import chunk_by_title
from unstructured.documents.elements import Element
from unstructured.partition.auto import partition

from clients import BaseClient, TransformersClient


class Chunk:
    text: str = ""
    metadata: dict = {}
    context: str = ""
    embeddings: List[float] = []

    def __init__(
            self,
            text: Optional[str] = None,
            metadata: Optional[Dict] = None,
            context: Optional[str] = None,
            embeddings: Optional[List[float]] = None
    ):
        if text:
            self.text = text
        if metadata:
            self.metadata = metadata
        if context:
            self.context = context
        if embeddings:
            self.embeddings = embeddings

    @staticmethod
    def from_element(element: Element):
        return Chunk(
            text=element.text,
            metadata=element.metadata.to_dict()
        )

    @staticmethod
    def from_dict(data: dict):
        return Chunk(
            text=data.get("text", None),
            metadata=data.get("metadata", None),
            context=data.get("context", None),
            embeddings=data.get("embeddings", None)
        )

    def to_dict(self) -> Dict:
        return dict(
            text=self.text,
            metadata=self.metadata,
            context=self.context,
            embeddings=self.embeddings
        )

    def to_string(self) -> str:
        return f"Text: {self.text}\nContext: {self.context}"


def load_and_chunk(
        filepath: str,
        max_characters: Optional[int] = 512,
        overlap: Optional[int] = 20,
        strategy: Optional[str] = "title"
) -> List[Chunk]:
    """
    Loads the corpus from the provided file path and chunk it into smaller parts.
    """
    # parse document
    corpus = partition(filename=filepath)

    # chunk document
    if strategy == "title":
        corpus = chunk_by_title(elements=corpus, max_characters=max_characters, overlap=overlap)
    else:
        corpus = chunk_elements(elements=corpus, max_characters=max_characters, overlap=overlap)

    # cast to JSON-serializable object
    corpus = [Chunk.from_element(element=c) for c in corpus]

    return corpus


def augment(
        chunks: List[Chunk],
        client: TransformersClient
) -> List[Chunk]:
    """
    Improves the chunks by adding context from associated document.

    This method uses the technique describe `here <https://www.anthropic.com/news/contextual-retrieval>`_.
    """
    instructions = """
Consider the following document:

<document>

Provide a short context describing how the following chunk locates in the above document:

<chunk>

Answer only with the succinct context and nothing else, without introduction nor explanation.
"""

    # build full document
    document = "\n".join([c.text for c in chunks])

    # caching common part
    prompt_cache = DynamicCache()
    initial_prompt = instructions.split("<chunk>")[0].replace("<document>", document)
    inputs = client.pipe.tokenizer(initial_prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        prompt_cache = client.pipe.model(
            **inputs,
            past_key_values=prompt_cache
        ).past_key_values  # this is the common prompt cached

    corpus: List[Chunk] = []
    for chunk in chunks:
        # initialize new entry
        doc = Chunk(text=chunk.text, metadata=chunk.metadata)

        # build full prompt
        prompt = (
            instructions
            .replace("<document>", document)
            .replace("<chunk>", chunk.text)
        )

        # compute chunk context in the document
        new_inputs = client.pipe.tokenizer(prompt, return_tensors="pt").to("cuda")
        past_key_values = copy.deepcopy(prompt_cache)
        outputs = client.pipe.model.generate(**new_inputs, past_key_values=past_key_values, max_new_tokens=256)
        doc.context = client.pipe.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0][len(prompt):]

        corpus.append(doc)

    return corpus


def embed(
        chunks: List[Chunk],
        client: BaseClient,
) -> List[Chunk]:
    """
    Computes the embeddings of the provided chunks.
    """
    embeddings = client.embed(texts=[c.to_string() for c in chunks])
    for i, chunk in enumerate(chunks):
        chunk.embeddings = embeddings[i].tolist()
    return chunks
