import argparse
from argparse import ArgumentParser

from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredPDFLoader as PDFLoader, \
    UnstructuredWordDocumentLoader as WordLoader, UnstructuredHTMLLoader as HTMLLoader
from langchain_community.vectorstores import FAISS
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

from revito.llm import OpenAILLM
from revito.prompts import CONTEXT_TEACHER, CONTEXT_ASKER


def load_parameters() -> argparse.Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="input file containing the lecture to revise",
    )
    return parser.parse_args()


def build_retriever(
        filepath: str,
):
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
        data = SemanticChunker(OpenAIEmbeddings()).split_documents(data)
        data = [d for d in data]
    except Exception as error:
        raise Exception(f"### ERROR (splitting): {error}")

    # compute vector store
    try:
        retriever = FAISS.from_documents(data, OpenAIEmbeddings()).as_retriever()
    except Exception as error:
        raise Exception(f"### ERROR (embedding): {error}")

    return retriever


def main(
        parameters: argparse.Namespace
):
    # load API keys
    load_dotenv()

    # build retriever from document
    retriever = build_retriever(filepath=parameters.input)

    # build LLM
    partner = OpenAILLM(context=CONTEXT_ASKER)
    teacher = OpenAILLM(context=CONTEXT_TEACHER)

    # start conversation
    exit_required = False
    while not exit_required:
        description = input("Décrivez la partie du cours que vous souhaitez réviser.\n>>> ").strip()
        if description.lower() in ["exit", "quit", "q"]:
            exit_required = True
        elif description is None:
            print("Veuillez fournir une description valide.")
        elif len(description) == 0:
            print("Veuillez fournir une description valide.")
        else:
            # get relevant part of lecture
            data = retriever.get_relevant_documents(description)
            partner.data = "\n\n".join([d.page_content for d in data])
            teacher.data = "\n\n".join([d.page_content for d in data])

            # provide question
            question = partner.respond("Pose-moi une question sur le cours.")

            # get user's response
            response = input(f"Question: {question}\nRéponse: ")

            # provide evaluation
            evaluation = teacher.respond(response)
            print(f"Evaluation: {evaluation}")


if __name__ == "__main__":
    params = load_parameters()
    main(parameters=params)
