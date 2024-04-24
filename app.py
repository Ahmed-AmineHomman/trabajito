import argparse
import os
from argparse import ArgumentParser
from typing import Tuple, Any, Optional

import gradio as gr
from dotenv import load_dotenv
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import FAISS

from utils import LLM, split_corpus, load_corpus

APP_NAME = "Revito"
APP_DESCRIPTION = """
Revito est une application d'aide à la révision utilisant les IAs conversationnelles pour vous aider dans vos révisions.
Elle combine l'utilisation de deux IAs :

- Le **partenaire** : l'IA chargée de vous poser des questions à propos du contenu des documents fournis à l'application, sur la thématique de votre choix.
- Le **correcteur** : l'IA chargée d'évaluer votre réponse.

Pour commencer, chargez le document de votre choix dans l'application en appuyant sur le bouton "Charger".
Une fois le chargement effectué, décrivez la thématique que vous souhaitez réviser puis appuyer sur le bouton "Envoyer".
Les IAs se mettront alors en marche et vous pourrez entamer le cycle de questions/réponses/corrections.

Bonnes révisions !
"""

TEACHER: LLM
PARTNER: LLM

SYSTEM_TEACHER = """
Evalue la qualité de la réponse de l'utilisateur en prenant en compte ce qui se trouve dans DATA.
Fournis une note de 1 à 5 (sois strict), que tu justifieras.
Prends soin de toujours donner la bonne réponse après ton évaluation.
"""
SYSTEM_PARTNER = """
A chaque demande de l'utilisateur, formules une question à l'utilisateur à propos de ce qui se trouve dans DATA.
Formules des questions courtes.
Réponds uniquement avec ta question, rien d'autre.
Formules une seule question à chaque demande.
"""


def load_parameters() -> argparse.Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--api-key",
        required=False,
        type=str,
        help="token corresponding to the API chosen with the '--api' parameter"
    )
    return parser.parse_args()


def build_retriever(
        filepath: Optional[str],
) -> Tuple[Any, str]:
    """Builds the retriever associated with the provided corpus."""
    # load corpus
    if not filepath:
        gr.Warning("aucun fichier n'a été fourni -> aucune opération effectuée")
        return None
    gr.Info("Chargement du corpus...")
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

    gr.Info("Corpus chargé.")
    return retriever, filepath


def set_theme(
        theme: str,
        retriever
) -> str:
    # collect relevant data
    try:
        data = "\n\n".join([d.page_content for d in retriever.get_relevant_documents(theme)])
    except Exception as error:
        raise gr.Error(f"### ERROR (corpus build): {error}")

    # reset llm data
    TEACHER.reset()
    PARTNER.reset()
    TEACHER.data = data
    PARTNER.data = data

    gr.Info("thème défini")


def get_new_question() -> str:
    """Formulate a new question from the internal data of the PARTNER."""
    try:
        question = PARTNER.respond("Pose-moi une question sur le cours", temperature=0.7)
    except Exception as error:
        raise gr.Error(f"### ERROR (question): {error}")
    return question


def evaluate_response(
        question: str,
        response: str,
) -> Tuple[str, str]:
    """Provide evaluation to the user's response as well as a new question."""
    try:
        evaluation = TEACHER.respond(query=f"Question: {question}. User response: {response}", temperature=0.1)
    except Exception as error:
        raise gr.Error(f"### ERROR (evaluation): {error}")
    return evaluation, response


if __name__ == "__main__":
    params = load_parameters()

    # load api keys
    load_dotenv()
    if params.api_key:
        os.environ["COHERE_API_KEY"] = params.api_key

    # initialize llms
    TEACHER = LLM(system=SYSTEM_TEACHER, api_key=params.api_key)
    PARTNER = LLM(system=SYSTEM_PARTNER, api_key=params.api_key)

    # build ui
    with gr.Blocks(title="Revito") as app:
        # define UI elements
        gr.Markdown(f"# {APP_NAME}\n\n{APP_DESCRIPTION}")
        with gr.Row():
            corpus_btn = gr.UploadButton("Charger", scale=1)
            filepath_text = gr.Textbox(label="Fichier", placeholder="chemin du fichier", interactive=False, scale=3)
        with gr.Row():
            theme_input = gr.Textbox(label="Thématique", placeholder="Décrivez la thématique à réviser ici...", scale=3)
            theme_btn = gr.Button("Envoyer", scale=1)
        with gr.Row():
            question_txt = gr.Textbox(label="Question", interactive=False)
            response_txt = gr.Textbox(label="Votre Réponse", interactive=False)
            evaluation_txt = gr.Textbox(label="Correction", interactive=False)
        with gr.Row():
            user_input = gr.Textbox(label="Réponse", placeholder="Répondez ici à la question", scale=3)
            user_btn = gr.Button("Répondre", scale=1)
            new_question_btn = gr.Button("Nouvelle question", scale=1)
            clear_btn = gr.ClearButton(components=[], value="Réinitialiser", scale=1)
        retriever = gr.State(value=None)

        # define UI logic
        corpus_btn.upload(
            fn=build_retriever,
            inputs=[corpus_btn],
            outputs=[retriever, filepath_text],
        )
        theme_input.submit(
            fn=set_theme,
            inputs=[theme_input, retriever],
            outputs=[]
        )
        theme_btn.click(  # this is a duplicate of `theme_desc.submit`
            fn=set_theme,
            inputs=[theme_input, retriever],
            outputs=[]
        )
        new_question_btn.click(
            fn=get_new_question,
            inputs=[],
            outputs=[question_txt]
        )
        user_input.submit(
            fn=evaluate_response,
            inputs=[question_txt, user_input],
            outputs=[evaluation_txt, response_txt]
        )
        user_btn.click(  # this is a duplicate of `user_input.submit`
            fn=evaluate_response,
            inputs=[question_txt, user_input],
            outputs=[evaluation_txt, response_txt]
        )

    app.launch(share=False)
