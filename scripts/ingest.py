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
    python ingest.py --files <file1> <file2> ... [--database <path_to_database>]

Arguments:
    --files: List of files to process.
    --database: Path to the database containing the embeddings (default: ./data/embeddings.db).
"""
import logging
import os
from argparse import ArgumentParser, Namespace
from pathlib import Path

from api.clients import TransformersClient, SentenceTransformerClient
from api.database import Database


def load_parameters() -> Namespace:
    """Loads the script parameters"""
    parser = ArgumentParser(
        description="Process the provided documents to create a JSON file containing augmented text contents. "
                    "The script reads and partitions the content of each file into chunks, cleans and augments each chunk, "
                    "and saves the processed chunks into a JSON file for creating an embeddings database."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="The files to process."
    )
    parser.add_argument(
        "--output",
        required=False,
        default="./data/database.jsonl",
        help="Path where the database will be exported."
    )
    parser.add_argument(
        "--llm_id",
        required=False,
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="The deposit id of the Hugging Face hub corresponding to the LLM performing the augmentation."
    )
    parser.add_argument(
        "--encoder_id",
        required=False,
        default="nomic-ai/modernbert-embed-base",
        help="The deposit id of the model in the sentence_transformers library computing the text embeddings."
    )
    parser.add_argument(
        "--skip_augmentation",
        action="store_true",
        help="Skip the augmentation step."
    )
    parser.add_argument(
        "--skip_embedding",
        action="store_true",
        help="Skip the embedding step."
    )
    return parser.parse_args()


def main(parameters):
    """Applies the text processing pipeline."""
    logging.info("initializing database...")
    output_path = Path(parameters.output).as_posix()
    database = Database()
    database.dump(output_path=output_path, reset_output=True)

    logging.info("loading llm...")
    if not parameters.skip_augmentation:
        llm = TransformersClient(model=parameters.llm_id)
    if not parameters.skip_embedding:
        encoder = SentenceTransformerClient(model=parameters.encoder_id)

    for file in parameters.inputs:
        logging.info(f"ingesting {file}")
        if not os.path.exists(file):
            message = f"File not found: {file}"
            logging.warning(message)
            pass
        filepath = Path(file).as_posix()

        logging.info("chunking file")
        database.ingest(filepath=filepath)

        logging.info("apply processing pipeline")
        database.process(filepath=filepath)

        if not parameters.skip_augmentation:
            logging.info("augmenting chunks")
            database.augment(client=llm, filepath=filepath)

        if not parameters.skip_embedding:
            logging.info("embedding chunks")
            database.embed(client=encoder, filepath=filepath)

        logging.info("saving corpus...")
        database.dump(
            output_path=output_path,
            filepath=filepath,
            reset_output=False,
            indent=4,
            ensure_ascii=True,
        )


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
