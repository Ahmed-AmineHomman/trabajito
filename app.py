import argparse
import os
from argparse import ArgumentParser
from typing import Any, Optional, Dict, List, Tuple

import gradio as gr
import tomli
from dotenv import load_dotenv
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import FAISS

from api.clients import Cohere, ChatExchange
from api.utils import split_corpus, load_corpus

HELP: Dict[str, str]
TEACHER: Cohere


def load_parameters() -> argparse.Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--api-key",
        required=False,
        type=str,
        help="token corresponding to the API chosen with the '--api' parameter"
    )
    parser.add_argument(
        "--envpath",
        required=False,
        type=str,
        help="path to the.env file"
    )
    parser.add_argument(
        "--language",
        choices=["en"],
        default="en",
        help=""
    )
    return parser.parse_args()


def build_retriever(
        filepath: Optional[str],
) -> Any:
    """Builds the retriever associated with the provided corpus."""
    # load corpus
    if not filepath:
        gr.Warning("no file provided -> skipping corpus loading")
        return None
    data = load_corpus(filepath=filepath)

    # split corpus
    try:
        data = split_corpus(corpus=data, chunk_size=500, chunk_overlap=20)
    except Exception as error:
        raise gr.Error(f"### ERROR (splitting) : {error}")

    # compute vector store
    try:
        retriever = FAISS.from_documents(data, CohereEmbeddings(model="embed-multilingual-v3.0")).as_retriever()
    except Exception as error:
        raise gr.Error(f"### ERROR (embedding): {error}")

    gr.Info("Corpus loaded")
    return "corpus loaded", retriever


def set_theme(
        theme: str,
        retriever
) -> Tuple[Dict[str, str], List[str]]:
    gr.Info("Setting theme...")

    # collect relevant data
    corpus = retriever.get_relevant_documents(theme)

    # define collection
    data = {f"chunk{i}": d.page_content for i, d in enumerate(corpus)}

    gr.Info("Theme set")
    return data, []


def ask_question(
        data: Dict[str, str],
        previous_questions: List[str]
) -> Tuple[str, List[str]]:
    """Formulates a question to the user."""
    # define system prompt
    system_prompt = f"{HELP.get('SYSTEM_PROMPT')}"
    system_prompt += "\n\nQuestions already asked:\n"
    system_prompt += "\n".join(previous_questions)

    # get question
    question = TEACHER.respond(
        query=HELP.get("FIRST_QUERY"),
        system_prompt=system_prompt,
        conversation=[],
        data=data,
        temperature=0.5
    )

    # add question to previous questions
    previous_questions.append(question)

    return question, previous_questions


def evaluate_response(
        question: str,
        response: str,
        data: Dict[str, str]
) -> str:
    """Provide evaluation to the user's response as well as a new question."""
    return TEACHER.respond(
        query=response,
        system_prompt=HELP.get("SYSTEM_PROMPT"),
        conversation=[ChatExchange(query=HELP.get("FIRST_QUERY"), response=question)],
        data=data,
        temperature=0.0
    )


def build_ui() -> gr.Blocks:
    with gr.Blocks(title=HELP.get("APP_NAME")) as app:
        # define UI elements
        gr.Markdown(f"# {HELP.get('APP_NAME')}\n\n{HELP.get('APP_DESCRIPTION')}")
        with gr.Accordion(label="Corpus"):
            with gr.Row():
                corpus_btn = gr.UploadButton("Corpus", scale=1)
                corpus_status = gr.Text(value=None, interactive=False, scale=3)
            with gr.Row():
                theme_btn = gr.Button("Charger", scale=1)
                theme = gr.Text(
                    label="Thématique",
                    placeholder="Décrivez la thématique à réviser ici...",
                    scale=3
                )
        with gr.Row():
            new_question_btn = gr.Button(value="Nouvelle question", variant="primary", scale=1, )
            question = gr.Text(label="Question", placeholder="Décrivez la question ici...", scale=3)
        with gr.Row():
            answer_btn = gr.Button(value="Répondre", variant="primary", scale=1)
            answer = gr.Textbox(label="Réponse", placeholder="Répondez ici à la question", scale=3)
        evaluation = gr.TextArea(label="Correction", interactive=False)
        retriever = gr.State(value=None)
        data = gr.State(value={})
        previous_questions = gr.State(value=[])

        # define UI logic
        corpus_btn.upload(
            fn=build_retriever,
            inputs=[corpus_btn],
            outputs=[corpus_status, retriever],
        )
        theme_btn.click(
            fn=set_theme,
            inputs=[theme, retriever],
            outputs=[data, previous_questions]
        )
        new_question_btn.click(
            fn=ask_question,
            inputs=[data, previous_questions],
            outputs=[question, previous_questions]
        )
        answer_btn.click(
            fn=evaluate_response,
            inputs=[question, answer, data],
            outputs=[evaluation]
        )
    return app


if __name__ == "__main__":
    params = load_parameters()

    # load api keys
    load_dotenv()
    if params.api_key:
        os.environ["COHERE_API_KEY"] = params.api_key

    # load UI doc
    with open(os.path.join("config", f"{params.language}.toml"), "rb") as f:
        HELP = tomli.load(f)

    # initialize llms
    TEACHER = Cohere()

    # run app
    build_ui().launch(share=False)
