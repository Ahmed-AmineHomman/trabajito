import argparse
import os
from argparse import ArgumentParser
from typing import Dict

import gradio as gr
import tomli

from api.clients import CohereClient
from app_api import build_retriever, set_theme, ask_question, evaluate_response

HELP: Dict[str, str]
TEACHER: CohereClient


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
    if params.api_key:
        os.environ["COHERE_API_KEY"] = params.api_key

    # load UI doc
    with open(os.path.join("config", f"{params.language}.toml"), "rb") as f:
        HELP = tomli.load(f)

    # initialize llms
    TEACHER = CohereClient()

    # run app
    build_ui().launch(share=False)
