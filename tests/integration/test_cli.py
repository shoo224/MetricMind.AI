import pytest

from metricmind.config.settings import get_settings
from metricmind.interfaces.cli.commands import run_command
from metricmind.main import main


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_RAW_DIR", str(tmp_path / "no_such_raw_dir"))
    monkeypatch.setenv("DATA_PROCESSED_DIR", str(tmp_path / "processed"))
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("REPORT_DIR", str(tmp_path / "reports"))
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_missing_dataset_gives_clear_error_and_exit_code_1(capsys):
    assert run_command("prepare", []) == 1
    err = capsys.readouterr().err
    assert "DataValidationError" in err and "not found" in err


def test_evaluate_without_training_explains_what_to_do(capsys):
    assert run_command("evaluate", []) == 1
    assert "train_model.py" in capsys.readouterr().err


def test_unknown_command_prints_usage(capsys):
    assert main(["bogus"]) == 2
    assert "usage" in capsys.readouterr().err
