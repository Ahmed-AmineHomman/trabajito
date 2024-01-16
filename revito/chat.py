from enum import Enum
from typing import Optional

from langchain_community.document_loaders import UnstructuredPDFLoader as PDFLoader, \
    UnstructuredWordDocumentLoader as WordLoader, UnstructuredHTMLLoader as HTMLLoader
from langchain_community.vectorstores import FAISS
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

from revito.llm import OpenAILLM
from .prompts import set_prompts


class ChatCommands(Enum):
    QUIT = -1
    CONTINUE = 0
    CHANGE_SECTION = 1


class ChatStatus(Enum):
    QUIT = -1
    CONTINUE = 1


class ChatbotManager:
    _status = ChatCommands.CHANGE_SECTION

    def __init__(
            self,
            filepath: str,
            prompts: Optional[dict] = None,
            api_key: Optional[str] = None,
    ):
        # build retriever from document
        self.retriever = _build_retriever(filepath=filepath)

        # build LLMs
        self._prompts = set_prompts() if prompts is None else prompts
        self._partner = OpenAILLM(context=self._prompts["partner_context"], api_key=api_key)
        self._teacher = OpenAILLM(context=self._prompts["teacher_context"], api_key=api_key)

    def interpret(
            self,
            command: Optional[str] = None,
    ) -> ChatStatus:
        """Interprets the user's command."""
        # interpret user command
        if command in ["exit", "quit", "q"]:
            self._status = ChatCommands.QUIT
        elif command == "change":
            self._status = ChatCommands.CHANGE_SECTION
        else:
            pass

        # act accordingly
        return self._act()

    def _act(self) -> ChatStatus:
        """Performs the action corresponding to the status."""
        if self._status == ChatCommands.CONTINUE:
            question = self._partner.respond(self._prompts["revision_query"])
            response = input(self._prompts["revision_builder"](question))
            evaluation = self._teacher.respond(response)
            print(f"Evaluation: {evaluation}")
            return ChatStatus.CONTINUE
        elif self._status == ChatCommands.CHANGE_SECTION:
            description = input(self._prompts["section_input"]).strip()

            # get relevant part of lecture
            data = self.retriever.get_relevant_documents(description)
            self._partner.data = "\n\n".join([d.page_content for d in data])
            self._teacher.data = "\n\n".join([d.page_content for d in data])

            # update status
            self._status = ChatCommands.CONTINUE
            return ChatStatus.CONTINUE
        elif self._status == ChatCommands.QUIT:
            return ChatStatus.QUIT
        else:
            raise ValueError(f"### Error : unknown status {self._status}")


def _build_retriever(
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
