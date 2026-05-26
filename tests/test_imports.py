"""CPU import smoke tests.

Verify the core modules import cleanly. Modules that pull in `arnie`
(Dataset, Env, Functions, run) are tested behind a skipif because
arnie is an external lab package that may not be present.
"""
import importlib

import pytest

ARNIE_AVAILABLE = importlib.util.find_spec("arnie") is not None
skip_if_no_arnie = pytest.mark.skipif(
    not ARNIE_AVAILABLE, reason="arnie not installed; see README installation notes"
)


def test_dropout_imports():
    import dropout  # noqa: F401


def test_encoder_decoder_imports():
    import Encoder_Decoder  # noqa: F401


def test_network_test10_imports():
    import Network_test10  # noqa: F401


@skip_if_no_arnie
def test_dataset_imports():
    import Dataset  # noqa: F401


@skip_if_no_arnie
def test_env_imports():
    import Env  # noqa: F401


@skip_if_no_arnie
def test_functions_imports():
    import Functions  # noqa: F401
