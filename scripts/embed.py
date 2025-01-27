"""
Embeds the provided JSON database and outputs a JSONL file with the embeddings.
"""
import json
import logging
import os
from argparse import ArgumentParser, Namespace
from pathlib import Path

from sentence_transformers import SentenceTransformer

from api.utils import Chunk


def load_parameters() -> Namespace:
    parser = ArgumentParser(
        description="Embeds the provided JSON database and outputs a JSONL file with the embeddings."
    )
    parser.add_argument(
        "--files",
        nargs="+",
        required=True,
        type=str,
        help="The files to process."
    )
    parser.add_argument(
        "--output",
        required=False,
        type=str,
        default=".data/",
        help="Directory where embeddings will be saved."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="nomic-ai/modernbert-embed-base",
        help="identifier of the sentence-transformer model to use for the embeddings."
    )
    return parser.parse_args()


def main(parameters):
    logging.info("loading model")
    model = SentenceTransformer(parameters.model)

    for file in parameters.files:
        logging.info(f"processing file {file}")
        if not os.path.exists(file):
            logging.error(f"File {file} does not exist.")
            continue
        filepath = Path(file)

        logging.info("loading corpus from JSON file")
        with open(filepath, "rb") as f:
            corpus = json.load(fp=f)
        corpus = [Chunk.from_dict(d) for d in corpus]

        logging.info("aggregate corpus into text chunks")
        corpus = [c.to_string() for c in corpus]

        logging.info("compute embeddings")
        embeddings = model.encode(corpus)

        logging.info("saving embeddings")
        with open(os.path.join(parameters.output, f"{filepath.stem}.jsonl"), "w") as f:
            for text, emb in zip(corpus, embeddings):
                json.dump({"text": text, "embedding": emb.tolist()}, f)
                f.write("\n")


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
