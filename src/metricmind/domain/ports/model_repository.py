"""ModelRepository port: persist / reload trained artefacts so we never retrain on start-up."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class ModelRepository(ABC):
    @abstractmethod
    def save(self, name: str, artefact: object) -> Path:
        """Persist an arbitrary fitted object (detector, scaler) under ``name``."""

    @abstractmethod
    def load(self, name: str) -> Any:
        """Reload it. Raises ModelNotFoundError if absent."""

    @abstractmethod
    def save_metadata(self, name: str, metadata: dict[str, Any]) -> Path: ...

    @abstractmethod
    def load_metadata(self, name: str) -> dict[str, Any]: ...
