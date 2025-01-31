import copy
import json
from typing import List, Dict
from typing import Optional

import torch
from transformers import DynamicCache
from unstructured.chunking.basic import chunk_elements
from unstructured.chunking.title import chunk_by_title
from unstructured.cleaners.core import clean
from unstructured.documents.elements import Element
from unstructured.partition.auto import partition

from .clients import BaseClient, TransformersClient


class Chunk:
    text: str = ""
    context: str = ""
    source: str = ""
    source_summary: str = ""
    embeddings: List[float] = []

    def __init__(
            self,
            text: str,
            context: Optional[str] = None,
            source: Optional[str] = None,
            source_summary: Optional[str] = None,
            embeddings: Optional[List[float]] = None
    ):
        self.text = text
        if context:
            self.context = context
        if source:
            self.source = source
        if source_summary:
            self.source_summary = source_summary
        if embeddings:
            self.embeddings = embeddings

    @staticmethod
    def from_element(element: Element):
        source = f"file {element.metadata.filename}"
        if element.metadata.page_number:
            source += f" page {element.metadata.page_number}"
        if element.metadata.page_name:
            source += f" ({element.metadata.page_name})"
        return Chunk(text=element.text, source=source)

    @staticmethod
    def from_dict(data: dict):
        if "text" not in data.keys():
            raise ValueError("The provided data does not contain a 'text' key.")
        return Chunk(
            text=data.get("text"),
            context=data.get("context", None),
            source=data.get("source", None),
            source_summary=data.get("source_summary", None),
            embeddings=data.get("embeddings", None)
        )

    def to_dict(self) -> Dict:
        return dict(
            text=self.text,
            context=self.context,
            source=self.source,
            source_summary=self.source_summary,
            embeddings=self.embeddings
        )

    def to_string(self) -> str:
        output = self.text
        if len(self.context) > 0:
            output += f"\n\nContext: {self.context}"
        if len(self.source) > 0:
            output += f"\n\nSource: {self.source}"
        if len(self.source_summary) > 0:
            output += f"\n\nSource Summary: {self.source_summary}"
        return output


class Database:
    """
    A database responsible for storing search and document retrieval.
    """
    chunks: Dict[str, List[Chunk]] = {}

    _augment_template: str = """
Consider the following document:

<document>

Provide a short context description (a few sentences top) of how the following chunk locates in the overall document:

<chunk>

Answer only with the short description and nothing else, without introduction nor explanation.
"""

    def __init__(
            self,
            chunks: Optional[Dict[str, List[Chunk]]] = None
    ):
        if chunks:
            for key, values in chunks.items():
                self.chunks[key] = [d for d in values]

    def ingest(
            self,
            filepath: str,
            max_characters: Optional[int] = 512,
            overlap: Optional[int] = 20,
            strategy: Optional[str] = "title"
    ) -> None:
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

        # cast to Chunk objects
        corpus = [Chunk.from_element(element=c) for c in corpus]

        # add chunks to database
        if filepath in self.chunks.keys():
            self.chunks[filepath] += corpus
        else:
            self.chunks[filepath] = corpus

    def process(
            self,
            filepath: Optional[str] = None
    ) -> None:
        """
        Applies the text processing pipeline to the chunk's contents.
        """
        if filepath:
            if filepath not in self.chunks.keys():
                raise ValueError(f"File path {filepath} not found in the database.")
            else:
                files = [filepath]
        else:
            files = list(self.chunks.keys())

        for file in files:
            chunks = self.chunks[file]
            for chunk in chunks:
                text = chunk.text.strip()
                text = clean(text, lowercase=True, extra_whitespace=True, dashes=True, bullets=True)
                chunk.text = " ".join([w.strip() for w in text.split() if len(w) > 0])

    def augment(
            self,
            client: TransformersClient,
            filepath: Optional[str] = None
    ) -> None:
        """
        Improves the chunks by adding context from associated document.

        This method uses the technique describe `here <https://www.anthropic.com/news/contextual-retrieval>`_.
        """
        if filepath:
            if filepath not in self.chunks.keys():
                raise ValueError(f"File path {filepath} not found in the database.")
            else:
                files = [filepath]
        else:
            files = list(self.chunks.keys())

        for file in files:
            chunks = self.chunks[file]

            # build full document
            document = "\n".join([c.text for c in chunks])

            # caching common part
            prompt_cache = DynamicCache()
            initial_prompt = self._augment_template.split("<chunk>")[0].replace("<document>", document)
            inputs = client.pipe.tokenizer(initial_prompt, return_tensors="pt").to("cuda")
            with torch.no_grad():
                prompt_cache = client.pipe.model(
                    **inputs,
                    past_key_values=prompt_cache
                ).past_key_values  # this is the common prompt cached

            # augment chunks with context from whole document
            for chunk in chunks:
                # build full prompt
                prompt = (
                    self._augment_template
                    .replace("<document>", document)
                    .replace("<chunk>", chunk.text)
                )

                # compute chunk context in the document
                new_inputs = client.pipe.tokenizer(prompt, return_tensors="pt").to("cuda")
                past_key_values = copy.deepcopy(prompt_cache)
                outputs = client.pipe.model.generate(**new_inputs, past_key_values=past_key_values, max_new_tokens=256)
                context = client.pipe.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0][len(prompt):]

                # update entry
                chunk.context = context

    def embed(
            self,
            client: BaseClient,
            filepath: Optional[str] = None
    ) -> None:
        """
        Computes the embeddings of the provided chunks.
        """
        if filepath:
            if filepath not in self.chunks.keys():
                raise ValueError(f"File path {filepath} not found in the database.")
            else:
                files = [filepath]
        else:
            files = list(self.chunks.keys())

        for file in files:
            chunks = self.chunks[file]
            embeddings = client.embed(texts=[c.to_string() for c in chunks])
            for i, chunk in enumerate(chunks):
                chunk.embeddings = embeddings[i].tolist()

    def dump(
            self,
            output_path: str,
            filepath: Optional[str] = None,
            reset_output: Optional[bool] = False,
            **kwargs
    ) -> None:
        """
        Dumps the database to a JSONL file.
        """
        if filepath:
            if filepath not in self.chunks.keys():
                raise ValueError(f"File path {filepath} not found in the database.")
            else:
                files = [filepath]
        else:
            files = list(self.chunks.keys())
        if reset_output:
            with open(output_path, "w") as fh:
                pass

        for file in files:
            with open(output_path, "a") as fh:
                for chunk in self.chunks.get(file, []):
                    json.dump(chunk.to_dict(), fh, **kwargs)
                    fh.write("\n")
