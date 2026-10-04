"""Domain-level errors, so every layer can raise/catch them without knowing about pandas etc."""


class MetricMindError(Exception):
    """Base class for all project errors."""


class DataValidationError(MetricMindError):
    """The input data is malformed or violates a data-quality rule."""


class ModelNotFoundError(MetricMindError):
    """A requested persisted artefact (model, scaler, metadata) does not exist."""


class NotFittedError(MetricMindError):
    """A component was used before ``fit`` was called."""
