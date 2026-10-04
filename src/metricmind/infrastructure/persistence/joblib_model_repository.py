"""JoblibModelRepository: saves fitted objects with joblib and metadata as JSON."""

from __future__ import annotations

import json
import logging
import platform
from pathlib import Path
from typing import Any

import joblib
import numpy
import pandas
import sklearn

from metricmind.domain.exceptions import ModelNotFoundError
from metricmind.domain.ports.model_repository import ModelRepository

logger = logging.getLogger(__name__)


class JoblibModelRepository(ModelRepository):
    def __init__(self, model_dir: Path) -> None:
        self._dir = model_dir

    def save(self, name: str, artefact: object) -> Path:
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self._dir / f"{name}.joblib"
        joblib.dump(artefact, path)
        logger.info("Saved artefact %s", path)
        return path

    def load(self, name: str) -> Any:
        path = self._dir / f"{name}.joblib"
        if not path.is_file():
            raise ModelNotFoundError(f"{path} not found - run scripts/train_model.py first")
        return joblib.load(path)

    def save_metadata(self, name: str, metadata: dict[str, Any]) -> Path:
        self._dir.mkdir(parents=True, exist_ok=True)
        doc = {
            **metadata,
            "library_versions": {  # pickles are only safe to reload with compatible versions
                "python": platform.python_version(),
                "scikit-learn": sklearn.__version__,
                "pandas": pandas.__version__,
                "numpy": numpy.__version__,
            },
        }
        path = self._dir / f"{name}.json"
        path.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
        return path

    def load_metadata(self, name: str) -> dict[str, Any]:
        path = self._dir / f"{name}.json"
        if not path.is_file():
            raise ModelNotFoundError(f"{path} not found - run scripts/train_model.py first")
        return json.loads(path.read_text(encoding="utf-8"))
