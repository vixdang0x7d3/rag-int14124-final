from abc import ABC, abstractmethod


class BaseChunker(ABC):
    @abstractmethod
    def split_text(self, text: str) -> list[str]: ...
