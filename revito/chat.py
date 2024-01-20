from enum import Enum
from typing import Optional

from langchain_community.document_loaders import (
    UnstructuredPDFLoader as PDFLoader,
    UnstructuredWordDocumentLoader as WordLoader,
    UnstructuredHTMLLoader as HTMLLoader,
    TextLoader
)
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from .prompts import set_prompts


class ChatCommands(Enum):
    QUIT = -1
    CONTINUE = 0
    CHANGE_SECTION = 1


class ChatStatus(Enum):
    QUIT = -1
    CONTINUE = 1


class ChatbotManager:
    _command = ChatCommands.CHANGE_SECTION

    def __init__(
            self,
            filepath: str,
            prompts: Optional[dict] = None,
            api_key: Optional[str] = None,
    ):
        self._conversation = []
        self._system_texts = set_prompts() if prompts is None else prompts
        self._corpus: str = ""
        self._corpus_description: str = ""

        # build retriever from document
        self.retriever = _build_retriever(filepath=filepath)

        # set app prompts & texts
        self.partner_prompt = ChatPromptTemplate.from_messages(
            messages=[
                ("system", self._system_texts["partner"]["context_builder"]("{context}", "{data}")),
                ("user", self._system_texts["partner"]["query_builder"]("{query}")),
            ]
        )
        self.teacher_prompt = ChatPromptTemplate.from_messages(
            messages=[
                ("system", self._system_texts["teacher"]["context_builder"]("{context}", "{data}")),
                ("system", self._system_texts["teacher"]["query_builder"]("{query}")),
            ]
        )

        # set LLMs
        self.partner_llm = ChatOpenAI(model="gpt-3.5-turbo")
        self.teacher_llm = ChatOpenAI(model="gpt-3.5-turbo")

        # set up langchains!
        self.partner_chain = self.partner_prompt | self.partner_llm | StrOutputParser()
        self.teacher_chain = self.teacher_prompt | self.teacher_llm | StrOutputParser()

    def interpret(
            self,
            command: Optional[str] = None,
    ) -> ChatStatus:
        """Interprets the user's command."""
        # interpret user command
        if command in ["exit", "quit", "q"]:
            self._command = ChatCommands.QUIT
        elif command in ["change", "c"]:
            self._command = ChatCommands.CHANGE_SECTION
        else:
            pass

        # act accordingly
        return self._act()

    def _act(self) -> ChatStatus:
        """Performs the action corresponding to the status."""
        if self._command == ChatCommands.CONTINUE:
            # formulate question
            question = self.partner_chain.invoke({
                "context": self._system_texts["partner"]["context"],
                "data": self._corpus,
                "query": self._corpus_description,
            })
            print(f"Question: {question}")

            # get user's response
            response = input(self._system_texts["teacher"]["input"])

            # evaluate response
            evaluation = self.teacher_chain.invoke({
                "context": self._system_texts["teacher"]["context"],
                "data": question,
                "query": response,
            })
            print(f"Evaluation: {evaluation}")

            return ChatStatus.CONTINUE
        elif self._command == ChatCommands.CHANGE_SECTION:
            # ask user for corpus thematic
            self._corpus_description = input(self._system_texts["partner"]["input"])

            # get relevant chunks from corpus
            data = self.retriever.get_relevant_documents(self._corpus_description)
            self._corpus = "\n\n".join([d.page_content for d in data])

            # update status
            self._command = ChatCommands.CONTINUE

            return ChatStatus.CONTINUE
        elif self._command == ChatCommands.QUIT:
            return ChatStatus.QUIT
        else:
            raise ValueError(f"### Error : unknown status {self._command}")


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
        elif extension in ["htm", "html"]:
            data = HTMLLoader(filepath).load()
        elif extension in ["txt", "md", "rst"]:
            data = TextLoader(filepath).load()
        else:
            raise ValueError("Unsupported file type")
    except Exception as error:
        raise Exception(f"### ERROR (loading): {error}")

    # split text
    try:
        data = SemanticChunker(OpenAIEmbeddings()).split_documents(data)
        # data = [d for d in data]
    except Exception as error:
        raise Exception(f"### ERROR (splitting): {error}")

    # compute vector store
    try:
        retriever = (
            FAISS
            .from_documents(data, OpenAIEmbeddings())
            .as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "score_threshold": 0.2,
                    "k": 5,
                },
            )
        )
    except Exception as error:
        raise Exception(f"### ERROR (embedding): {error}")

    return retriever
