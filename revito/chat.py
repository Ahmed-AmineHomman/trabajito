from enum import Enum
from typing import Optional

from revito.llm import BaseLLM


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
            retriever,
            partner: BaseLLM,
            teacher: BaseLLM,
    ):
        self.retriever = retriever
        self.partner = partner
        self.teacher = teacher

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
            question = self.partner.respond("Pose-moi une question sur le cours.")
            response = input(f"Question: {question}\nRéponse: ")
            evaluation = self.teacher.respond(response)
            print(f"Evaluation: {evaluation}")
            return ChatStatus.CONTINUE
        elif self._status == ChatCommands.CHANGE_SECTION:
            description = input("Décrivez la partie du cours que vous souhaitez réviser.\n>>> ").strip()

            # get relevant part of lecture
            data = self.retriever.get_relevant_documents(description)
            self.partner.data = "\n\n".join([d.page_content for d in data])
            self.teacher.data = "\n\n".join([d.page_content for d in data])

            # update status
            self._status = ChatCommands.CONTINUE
            return ChatStatus.CONTINUE
        elif self._status == ChatCommands.QUIT:
            return ChatStatus.QUIT
        else:
            raise ValueError(f"### Error : unknown status {self._status}")
