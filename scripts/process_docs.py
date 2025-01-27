"""
This script processes a set of provided files to create a JSON file containing the processed text content.
The resulting JSON file will be used to create an embeddings database at a later stage.

The script performs the following steps:
1. Loads the script parameters, including the list of files to process and the path to the embeddings database.
2. Reads and partitions the content of each file into chunks using the `unstructured <https://docs.unstructured.io/welcome>`_ library.
3. Cleans each chunk of text by removing extra whitespace, converting to lowercase, and other text normalization steps.
4. Augments each chunk by adding context from the associated document using Meta's `llama-3.2-3b <https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct>`_ language model.
5. Saves the processed and augmented chunks into a JSON file at the specified database path.

Dependencies:
- `unstructured` library for parsing and cleaning text content.
- `transformers` library for interacting with the language model.
- `torch` library for handling tensor operations.

Usage:
    python process_docs.py --files <file1> <file2> ... [--database <path_to_database>]

Arguments:
    --files: List of files to process.
    --database: Path to the database containing the embeddings (default: ./data/embeddings.db).
"""
import copy
import json
import logging
from argparse import ArgumentParser, Namespace
from typing import List, Dict

import torch
from transformers import DynamicCache
from unstructured.chunking.title import chunk_by_title
from unstructured.cleaners.core import clean
from unstructured.documents.elements import Element
from unstructured.partition.auto import partition

from api.clients import TransformersClient

DEPOSIT_ID = "meta-llama/Llama-3.2-3B-Instruct"


def load_parameters() -> Namespace:
    """Loads the script parameters"""
    parser = ArgumentParser(
        description="Process the provided documents to create a JSON file containing augmented text contents. "
                    "The script reads and partitions the content of each file into chunks, cleans and augments each chunk, "
                    "and saves the processed chunks into a JSON file for creating an embeddings database."
    )
    parser.add_argument(
        "--files",
        nargs="+",
        required=True,
        help="The files to process."
    )
    parser.add_argument(
        "--database",
        required=False,
        default="./docs.json",
        help="Path to the database containing the embeddings."
    )
    return parser.parse_args()


def process(input: str) -> str:
    """
    Applies the text processing pipeline to the provided text.
    """
    output = input.strip()
    output = clean(output, lowercase=True, extra_whitespace=True, dashes=True, bullets=True)
    output = " ".join([w.strip() for w in output.split() if len(w) > 0])
    return output


def augment(
        chunks: List[Element],
        client: TransformersClient
) -> List[Dict]:
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

    corpus: List[Dict[str, str]] = []
    for chunk in chunks:
        # initialize new entry
        doc = dict(content=chunk.text, metadata=chunk.metadata.to_dict())

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
        doc["context"] = client.pipe.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0][len(prompt):]

        corpus.append(doc)

    return corpus


def main(parameters):
    """Applies the text processing pipeline."""
    logging.info("loading corpus")
    corpus = {f: partition(filename=f) for f in parameters.files}

    logging.info("processing chunks")
    for file, docs in corpus.items():
        corpus[file] = chunk_by_title(elements=docs, max_characters=512, overlap=20)

    corpus = {f: chunks[:3] for f, chunks in corpus.items()}

    logging.info("loading llm...")
    client = TransformersClient()
    client.load(model_id=DEPOSIT_ID)

    # augment chunks
    for file, docs in corpus.items():
        logging.info("augmenting chunks for file: %s", file)
        corpus[file] = augment(chunks=docs, client=client)

    logging.info("saving corpus...")
    with open(parameters.database, "w") as f:
        json.dump(corpus, f, indent=4, ensure_ascii=True)


if __name__ == "__main__":
    args = load_parameters()

    # configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s]::%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler()
        ]
    )

    # run the main function
    main(args)
