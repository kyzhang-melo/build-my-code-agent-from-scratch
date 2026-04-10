from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _install_import_stubs() -> None:
    if "dotenv" not in sys.modules:
        dotenv_stub = types.ModuleType("dotenv")
        dotenv_stub.load_dotenv = lambda *args, **kwargs: True
        sys.modules["dotenv"] = dotenv_stub

    if "openai" not in sys.modules:
        openai_stub = types.ModuleType("openai")

        class DummyOpenAI:
            def __init__(self, *args, **kwargs):
                self.responses = types.SimpleNamespace(create=lambda **_: None)

        openai_stub.OpenAI = DummyOpenAI
        sys.modules["openai"] = openai_stub


@pytest.fixture(scope="session", autouse=True)
def _test_runtime_setup() -> None:
    _install_import_stubs()
    os.environ.setdefault("OPENROUTER_API_KEY", "test-key")
    os.environ.setdefault("OPENROUTER_BASE_URL", "https://example.com/v1")
    os.environ.setdefault("MODEL_ID", "test-model")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))


@pytest.fixture
def load_module():
    def _load(module_name: str, relative_path: str):
        module_path = ROOT / relative_path
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        sys.modules.pop(module_name, None)
        spec.loader.exec_module(module)
        return module

    return _load
