import argparse
import os
from argparse import ArgumentParser
from typing import Tuple, Any, Optional, List

import gradio as gr
import pandas as pd
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
Tu es un assistant IA aidant un professeur à évaluer les réponses de ses étudiants.
Le professeur, ton utilisateur, va te fournir une question à propos de ce qui se trouve dans DATA et des réponses à la question proposées par ses étudiants
Prends soin de toujours renvoyer uniquement l'évaluation demandée par le professeur.
Utilises toujours un langage courtois et neutre.
Prends soin de bien justifier toute affirmation ou évaluation que te demanderas le professeur.
Cites les éléments de DATA que tu utilises pour tes évaluations.
Renvoies une évaluation que le professeur pourra directement transmettre à l'étudiant.
"""
SYSTEM_PARTNER = """
Tu es un assistant IA aidant un étudiant, ton utilisateur, à réviser ses cours, décrits dans DATA.
A chacune de ses demandes, formules une question relative à ce qui se trouve dans DATA.
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
) -> Any:
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
    return retriever


def set_theme(
        theme: str,
        scores: pd.DataFrame,
        retriever
) -> Tuple[List[List[str]], str, pd.DataFrame]:
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

    # reset scores
    scores = pd.DataFrame({
        "indicateur": ["note", "# réponses", "moyenne"],
        "valeur": [0.0, 0.0, 0.0]
    }).astype({"indicateur": str, "valeur": float})

    gr.Info("thème défini")
    return [], data, scores


def evaluate_response(
        response: str,
        discussion: List[List[str]],
        scores: pd.DataFrame,
        data: str,
) -> Tuple[pd.DataFrame, str, List[List[str]]]:
    """Provide evaluation to the user's response as well as a new question."""
    # evaluate response to existing question
    if len(discussion) > 0:
        question = discussion[-1][1]
        try:
            # retrieve evaluation
            TEACHER.reset()
            TEACHER.data = data
            evaluation = TEACHER.respond(
                query=f"""
La question formulée est la suivante: {question}.
La réponse de mon étudiant est la suivante: {response}
Evalue la qualité de cette réponse s'il te plaît.
""".replace("\n", " ").strip(),
                temperature=0.1
            )
            grade = TEACHER.respond(
                query="""
En fonction de ton évaluation, donnes-moi la note, de 1 à 5, que tu donnerais à cette réponse.
Renvoies-moi uniquement la note, et rien d'autre s'il te plaît.
""".replace("\n", " ").strip(),
                temperature=0.1
            )

            # update scores
            scores = scores.astype({"indicateur": str, "valeur": float})
            grade = eval(grade)
            n = scores["valeur"].iloc[1]
            mu = scores["valeur"].iloc[2]
            scores["valeur"].iloc[0] = grade
            scores["valeur"].iloc[1] = n + 1
            scores["valeur"].iloc[2] = (n * mu + grade) / (n + 1)
        except Exception as error:
            raise gr.Error(f"### ERROR (evaluation): {error}")
    else:
        evaluation = ""

    # retrieve new question
    try:
        question = PARTNER.respond("Pose-moi une question sur le cours", temperature=0.7)
    except Exception as error:
        raise gr.Error(f"### ERROR (question): {error}")

    return scores, evaluation, discussion + [[response, question]]


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Revito") as app:
        # define UI elements
        gr.Markdown(f"# {APP_NAME}\n\n{APP_DESCRIPTION}")
        with gr.Accordion(label="Corpus & Paramètres"):
            with gr.Row():
                corpus_btn = gr.UploadButton("Corpus", scale=1)
                theme_btn = gr.Button("Charger", scale=1)
                theme = gr.Text(label="Thématique", placeholder="Décrivez la thématique à réviser ici...", scale=3)
        with gr.Row():
            with gr.Column(scale=3):
                chat = gr.Chatbot(label="Discussion")
                answer = gr.Textbox(label="Réponse", placeholder="Répondez ici à la question")
            with gr.Column(scale=1):
                scores = gr.DataFrame(
                    interactive=False,
                    headers=["indicateur", "valeur"],
                    row_count=3,
                    col_count=2,
                    datatype=["str", "number"],
                )
                answer_eval = gr.TextArea(label="Correction", interactive=False)
                answer_btn = gr.Button("Répondre", scale=1, variant="primary")
        retriever = gr.State(value=None)
        data = gr.State(value="")

        # define UI logic
        corpus_btn.upload(
            fn=build_retriever,
            inputs=[corpus_btn],
            outputs=[retriever],
        )
        theme_btn.click(
            fn=set_theme,
            inputs=[theme, scores, retriever],
            outputs=[chat, data, scores]
        )
        answer.submit(  # this is a duplicate of `answer_btn.click`
            fn=evaluate_response,
            inputs=[answer, chat, scores, data],
            outputs=[scores, answer_eval, chat]
        )
        answer_btn.click(
            fn=evaluate_response,
            inputs=[answer, chat, scores, data],
            outputs=[scores, answer_eval, chat]
        )
    return app


if __name__ == "__main__":
    params = load_parameters()

    # load api keys
    load_dotenv()
    if params.api_key:
        os.environ["COHERE_API_KEY"] = params.api_key

    # initialize llms
    TEACHER = LLM(system=SYSTEM_TEACHER, api_key=params.api_key)
    PARTNER = LLM(system=SYSTEM_PARTNER, api_key=params.api_key)

    # run app
    build_ui().launch(share=False)
