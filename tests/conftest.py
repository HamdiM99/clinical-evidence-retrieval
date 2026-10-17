from pathlib import Path

import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "mini"


@pytest.fixture
def mini_dir():
    return FIXTURE
