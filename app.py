import argparse
from argparse import ArgumentParser
from typing import Tuple, Any

import gradio as gr
from dotenv import load_dotenv
from langchain_cohere import CohereEmbeddings
from langchain_community.document_loaders import UnstructuredPDFLoader as PDFLoader, \
    UnstructuredWordDocumentLoader as WordLoader, UnstructuredHTMLLoader as HTMLLoader
from langchain_community.vectorstores import FAISS

from utils import LLM, split_corpus

TEACHER: LLM
PARTNER: LLM

SYSTEM_TEACHER = """
Evalue la qualité de la réponse de l'utilisateur en prenant en compte ce qui se trouve dans DATA.
Fournis une note de 1 à 5 (sois strict), que tu justifieras.
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
        "--api",
        type=str,
        choices=["openai", "cohere"],
        default="cohere",
        help="API providing the LLMs powering the assistant"
    )
    parser.add_argument(
        "--api-key",
        required=False,
        type=str,
        help="token corresponding to the API chosen with the '--api' parameter"
    )
    return parser.parse_args()


def build_retriever(
        filepath: str,
) -> Tuple[str, Any]:
    # load corpus
    extension = filepath.split(".")[-1].lower()
    try:
        if extension == "pdf":
            data = PDFLoader(filepath).load()
        elif extension in ["doc", "docx", "odt"]:
            data = WordLoader(filepath).load()
        elif extension == "html":
            data = HTMLLoader(filepath).load()
        else:
            raise ValueError("Unsupported file type")
    except Exception as error:
        return f"### ERROR (loading): {error}", None

    # split corpus
    try:
        data = split_corpus(corpus=data, chunk_size=500, chunk_overlap=20)
    except Exception as error:
        return f"### ERROR (splitting): {error}", None

    # compute vector store
    try:
        retriever = FAISS.from_documents(data, CohereEmbeddings(model="embed-multilingual-v3.0")).as_retriever()
    except Exception as error:
        return f"### ERROR (embedding): {error}", None

    return "cours intégré", retriever


def set_theme(
        theme: str,
        retriever
) -> str:
    # collect relevant data
    try:
        data = "\n\n".join([d.page_content for d in retriever.get_relevant_documents(theme)])
    except Exception as error:
        return f"### ERROR (corpus build): {error}"

    # reset llm data
    TEACHER.data = data
    PARTNER.data = data

    return "thème défini"


def get_new_question() -> Tuple[str, str]:
    """Formulate a new question from the internal data of the PARTNER."""
    try:
        question = PARTNER.respond("Pose-moi une question sur le cours", temperature=0.7)
    except Exception as error:
        return f"### ERROR (question): {error}", None
    return "question posée", question


def evaluate_response(
        question: str,
        response: str,
) -> Tuple[str, str]:
    """Provide evaluation to the user's response as well as a new question."""
    # evaluate user's response
    try:
        evaluation = TEACHER.respond(query=f"Question: {question}. User response: {response}", temperature=0.1)
    except Exception as error:
        return f"### ERROR (evaluation): {error}", None

    return "évaluation terminée", evaluation


if __name__ == "__main__":
    load_dotenv()
    params = load_parameters()
    TEACHER = LLM(system=SYSTEM_TEACHER)
    PARTNER = LLM(system=SYSTEM_PARTNER)

    with gr.Blocks() as app:
        # define UI elements
        gr.Markdown("""
        # Revito
        
        Bienvenue dans Revito, votre assistant de révision. Commencez par fournir votre cours à l'assistant, précisez la thématique que vous souhaitez révisez et c'est parti !
        """)
        with gr.Row():
            corpus_handler = gr.File(label="Déposez ici votre cours", file_count="single")
            corpus_btn = gr.Button("Charger")
            status = gr.Text(label="Status", placeholder="barre de statut")
        with gr.Row():
            theme_desc = gr.Textbox(label="Thématique", placeholder="Décrivez la thématique à réviser ici...")
            with gr.Column():
                theme_btn = gr.Button("Envoyer")
                clear_btn = gr.ClearButton(components=[])
                new_question_btn = gr.Button("Nouvelle question")
        question_txt = gr.Text(label="Question", placeholder="La question apparaîtra ici.")
        response_txt = gr.Textbox(label="Réponse", placeholder="Ecrivez votre réponse ici...")
        evaluation_txt = gr.Text(label="Évaluation", placeholder="La correction apparaîtra ici.")
        retriever = gr.State(value=None)

        # define UI logic
        corpus_btn.click(
            fn=build_retriever,
            inputs=[corpus_handler],
            outputs=[status, retriever],
        )
        theme_desc.submit(
            fn=set_theme,
            inputs=[theme_desc, retriever],
            outputs=[status]
        )
        theme_btn.click(  # this is a duplicate of `theme_desc.submit`
            fn=set_theme,
            inputs=[theme_desc, retriever],
            outputs=[status]
        )
        new_question_btn.click(
            fn=get_new_question,
            inputs=[],
            outputs=[status, question_txt]
        )
        response_txt.submit(
            fn=evaluate_response,
            inputs=[question_txt, response_txt],
            outputs=[status, evaluation_txt]
        )

    app.launch(share=False)
