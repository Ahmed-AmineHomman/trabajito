import argparse
import json
from argparse import ArgumentParser

from dotenv import load_dotenv

from revito.chat import ChatbotManager, ChatStatus
from revito.prompts import set_prompts


def load_parameters() -> argparse.Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="input file containing the lecture to revise",
    )
    parser.add_argument(
        "--language", "-l",
        type=str,
        required=False,
        choices=["fr", "en"],
        default="fr",
        help="chatbot language ('fr' or 'en')",
    )
    parser.add_argument(
        "--api-key", "-k",
        type=str,
        required=False,
        default=None,
        help="OpenAI API key (if none provided, will look for corresponding key in the '.env' file on the app folder)",
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
    prompts = set_prompts(language=params.language)

    header = welcome()
    print(header)

    # load API keys
    load_dotenv()

    # initialize manager
    manager = ChatbotManager(filepath=params.input, prompts=prompts, api_key=params.api_key)

    # start chatting
    status = ChatStatus.CONTINUE
    while status == ChatStatus.CONTINUE:
        prompt = prompts["app"]["command"]
        command = input(prompt)
        status = manager.interpret(command=command)

    print(prompts["app"]["goodbye"])
