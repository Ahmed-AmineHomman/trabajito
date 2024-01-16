import argparse
import json
from argparse import ArgumentParser

from dotenv import load_dotenv

from revito.chat import ChatbotManager, ChatStatus
from revito.llm import OpenAILLM
from revito.prompts import CONTEXT_TEACHER, CONTEXT_ASKER
from revito.utilities import build_retriever


def load_parameters() -> argparse.Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="input file containing the lecture to revise",
    )
    parser.add_argument(
        "--api-key", "-k",
        type=str,
        required=False,
        default=None,
        help="OpenAI API key",
    )
    return parser.parse_args()


def welcome() -> str:
    # get package info
    with open("package_info.json", "r") as f:
        package_info = json.load(f)
    name = package_info["name"].capitalize()
    author = package_info["author"]
    version = package_info["version"]

    # build welcome header
    title = f"{name}"
    subtitle = f"version {version}, by {author}"
    splitter = "=" * max(len(title), len(subtitle))
    return f"{splitter}\n{title}\n{splitter}\n{subtitle}\n{splitter}\n"


if __name__ == "__main__":
    params = load_parameters()

    header = welcome()
    print(header)

    # load API keys
    load_dotenv()

    # build retriever from document
    retriever = build_retriever(filepath=params.input)

    # build LLMs
    partner = OpenAILLM(context=CONTEXT_ASKER)
    teacher = OpenAILLM(context=CONTEXT_TEACHER)

    # initialize manager
    manager = ChatbotManager(retriever=retriever, partner=partner, teacher=teacher)

    # start chatting
    status = ChatStatus.CONTINUE
    while status == ChatStatus.CONTINUE:
        command = input("Entrez votre commande ('q' pour quitter l'application, 'c' pour changer de thèmatique, entrée pour continuer): ")
        status = manager.interpret(command=command)

    print("Au revoir!")
