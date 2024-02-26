import argparse
from argparse import ArgumentParser
from typing import List, Tuple, Any

import gradio as gr
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredPDFLoader as PDFLoader, \
    UnstructuredWordDocumentLoader as WordLoader, UnstructuredHTMLLoader as HTMLLoader
from langchain_community.embeddings import CohereEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_experimental.text_splitter import SemanticChunker

from revito.llm import LLMFactory, BaseLLM

FACTORY = LLMFactory()
TEACHER: BaseLLM
PARTNER: BaseLLM

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
        "--model",
        type=str,
        choices=["openai", "cohere"],
        default="cohere",
        help="Large language models powering the assistant"
    )
    return parser.parse_args()


def build_store(
        filepath: str,
) -> Tuple[str, Any]:
    # load data from document
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
        raise Exception(f"### ERROR (loading): {error}")

    # split text
    try:
        data = SemanticChunker(CohereEmbeddings()).split_documents(data)
        data = [d for d in data]
    except Exception as error:
        raise Exception(f"### ERROR (splitting): {error}")

    # compute vector store
    try:
        retriever = FAISS.from_documents(data, CohereEmbeddings()).as_retriever()
    except Exception as error:
        raise Exception(f"### ERROR (embedding): {error}")

    return "cours intégré", retriever


def start(
        theme: str,
        retriever,
        chat_partner: gr.Chatbot,
) -> Tuple[str, List[Tuple[str]]]:
    # collect relevant data
    data = "\n\n".join([d.page_content for d in retriever.get_relevant_documents(theme)])

    # reset llms
    TEACHER.reset_messages()
    PARTNER.reset_messages()
    TEACHER.data = data
    PARTNER.data = data

    # formulate question
    question = PARTNER.respond("Pose-moi une question sur le cours", temperature=0.5)
    chat_partner.append(("Pose-moi une question sur le cours", question))

    return "thème défini", chat_partner


def respond(
        query: str,
        chat_teacher: gr.Chatbot,
        chat_partner: gr.Chatbot,
) -> Tuple[str, List[List[str]], List[List[str]]]:
    """Provide evaluation to the user's response as well as a new question."""
    # retrieve partner's question
    question = PARTNER.get_conversation()[-1][1]

    # evaluate user's response
    evaluation = TEACHER.respond(query=f"Question: {question}. User response: {query}", temperature=0.0)
    chat_teacher.append((query, evaluation))

    # formulate new question
    question = PARTNER.respond("Pose-moi une question sur le cours", temperature=0.5)
    chat_partner.append((query, question))

    return "évaluation terminée", chat_partner, chat_teacher


if __name__ == "__main__":
    load_dotenv()
    params = load_parameters()
    TEACHER = FACTORY(model=params.model, system=SYSTEM_TEACHER)
    PARTNER = FACTORY(model=params.model, system=SYSTEM_PARTNER)

    with gr.Blocks() as app:
        gr.Markdown("""
        # Revito
        
        Bienvenue dans Revito, votre assistant de révision. Commencez par fournir votre cours à l'assistant, précisez la thématique que vous souhaitez révisez et c'est parti !
        """)
        status = gr.Text(label="Status", placeholder="barre de statut")
        with gr.Row():
            file_explorer = gr.File(label="Choisissez votre cours", file_count="single")
            load = gr.Button("Charger")
        with gr.Row():
            theme = gr.Textbox(label="Thématique", placeholder="Décrivez la thématique à réviser ici...")
            set_theme = gr.Button("Envoyer")
        with gr.Row():
            chat_partner = gr.Chatbot()
            chat_teacher = gr.Chatbot()
        with gr.Row():
            query = gr.Textbox(label="Question", placeholder="Posez votre question ici...")
            clear = gr.ClearButton(components=[query, chat_partner, chat_teacher])

        retriever = gr.State(value=None)
        load.click(
            fn=build_store,
            inputs=[file_explorer],
            outputs=[status, retriever],
        )
        set_theme.click(
            fn=start,
            inputs=[theme, retriever, chat_partner],
            outputs=[status, chat_partner]
        )
        query.submit(
            fn=respond,
            inputs=[query, chat_partner, chat_teacher],
            outputs=[status, chat_partner, chat_teacher]
        )

    app.launch(share=False)
