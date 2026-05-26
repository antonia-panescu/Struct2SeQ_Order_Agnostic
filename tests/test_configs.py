"""Config YAML smoke tests."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_KEYS = {
    "db_vocab_size",
    "rna_vocab_size",
    "embed_size",
    "nhead",
    "num_encoder_layers",
    "num_decoder_layers",
    "dropout",
    "n_episodes",
    "epochs_per_episode",
    "train_batch_size",
    "max_train_batch_size",
    "inference_batch_size",
    "sequence_length",
    "learning_rate",
}


def _load(name):
    return yaml.safe_load((ROOT / name).read_text())


def test_default_config_has_required_keys():
    cfg = _load("default_config.yaml")
    missing = REQUIRED_KEYS - set(cfg)
    assert not missing, f"default_config.yaml missing keys: {sorted(missing)}"


def test_brev_config_has_required_keys():
    cfg = _load("config_brev_8gpu.yaml")
    missing = REQUIRED_KEYS - set(cfg)
    assert not missing, f"config_brev_8gpu.yaml missing keys: {sorted(missing)}"
