from typing import Optional, Any, Tuple, Dict, List

import gradio as gr

from app import HELP, TEACHER
from clients import ChatExchange
from utils import load_corpus, split_corpus


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
