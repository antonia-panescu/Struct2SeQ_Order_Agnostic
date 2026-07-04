import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_ARNIEFILE = PROJECT_ROOT / "arnie_file.txt"
WEIGHTS_DIR = PROJECT_ROOT / "weights"

if "ARNIEFILE" not in os.environ:
    if not DEFAULT_ARNIEFILE.exists():
        DEFAULT_ARNIEFILE.write_text("linearpartition: .\nTMP: /tmp\n")
    os.environ["ARNIEFILE"] = str(DEFAULT_ARNIEFILE)

from arnie.pk_predictors import _hungarian
from arnie.utils import convert_dotbracket_to_bp_list
from Network_test10 import RibonanzaNet


def mask_diagonal(matrix, mask_value=0):
    matrix = matrix.copy()
    n = len(matrix)
    for i in range(n):
        for j in range(n):
            if abs(i - j) < 4:
                matrix[i][j] = mask_value
    return matrix


class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)
        self.entries = entries

    def print(self):
        print(self.entries)


def load_config_from_yaml(file_path):
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return Config(**config)


class finetuned_RibonanzaNet(RibonanzaNet):
    def __init__(self, config):
        config.dropout = 0.3
        super(finetuned_RibonanzaNet, self).__init__(config)

        self.dropout = nn.Dropout(0.0)
        self.ct_predictor = nn.Linear(64, 1)

    def forward(self, src):
        _, pairwise_features = self.get_embeddings(
            src, torch.ones_like(src).long().to(src.device)
        )

        pairwise_features = pairwise_features + pairwise_features.permute(
            0, 2, 1, 3
        )  # symmetrize

        output = self.ct_predictor(self.dropout(pairwise_features))  # predict

        return output.squeeze(-1)


class DQN_env:
    def __init__(
        self,
        rnet_weights=None,
        rnet_ss_weights=None,
        use_gpu=True,
        compile=False,
    ):
        def resolve_weight(path, filename):
            if path is None:
                return WEIGHTS_DIR / filename
            path = Path(path)
            return path if path.is_absolute() else PROJECT_ROOT / path

        def load_weight_file(path, label):
            try:
                return torch.load(path, map_location="cpu")
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    f"{label} weights were not found at {path}. "
                    "Download the RibonanzaNet weights and place them in "
                    f"{WEIGHTS_DIR}."
                ) from exc

        rnet_weights = resolve_weight(rnet_weights, "RibonanzaNet.pt")
        rnet_ss_weights = resolve_weight(rnet_ss_weights, "RibonanzaNet-SS.pt")

        model = finetuned_RibonanzaNet(
            load_config_from_yaml(PROJECT_ROOT / "test10_configs/pairwise.yaml")
        )
        model.load_state_dict(load_weight_file(rnet_ss_weights, "RibonanzaNet-SS"))

        reactivity_model = RibonanzaNet(
            load_config_from_yaml(PROJECT_ROOT / "test10_configs/pairwise.yaml")
        )
        reactivity_model.load_state_dict(load_weight_file(rnet_weights, "RibonanzaNet"))

        self.SS_model = model

        self.tokens = {nt: i for i, nt in enumerate("ACGU")}
        self.index_to_nt = {index: nt for nt, index in self.tokens.items()}

        self.reactivity_model = reactivity_model

        self.use_gpu = use_gpu

        if self.use_gpu:
            self.SS_model = self.SS_model.cuda()
            self.reactivity_model = self.reactivity_model.cuda()

        if compile:
            self.SS_model = torch.compile(self.SS_model)
            self.reactivity_model = torch.compile(self.reactivity_model)
        self.SS_model.eval()
        self.reactivity_model.eval()

    def get_SHAPE(self, src):
        with torch.no_grad():
            SHAPE = self.reactivity_model(src)[:, :, 0].cpu().numpy()
        return SHAPE

    def get_structure(self, src):

        with torch.no_grad():
            bpps = self.SS_model(src).sigmoid().cpu().numpy()  # .squeeze(-1)

        structures, bps = [], []
        for bpp in bpps:
            structure, bp = _hungarian(mask_diagonal(bpp), theta=0.5, min_len_helix=1)
            structures.append(structure)
            bps.append(bp)

        return structures, bps

    def get_paired_correspondences(self, bp):
        paired_correspondences = {}
        for i, j in bp:
            paired_correspondences[i] = j
            paired_correspondences[j] = i
        return paired_correspondences

    def get_reward(self, bps, target_structure):
        target_bp = convert_dotbracket_to_bp_list(
            target_structure, allow_pseudoknots=True
        )

        target_correspondence = self.get_paired_correspondences(target_bp)

        design_correspondence = self.get_paired_correspondences(bps)

        positional_rewards = []
        sum_rewards = 0
        for i in range(len(target_structure)):
            if i in target_correspondence and i in design_correspondence:
                if target_correspondence[i] == design_correspondence[i]:
                    pos_reward = 1
                    sum_rewards += 1
                else:
                    pos_reward = 0
            elif i not in target_correspondence and i not in design_correspondence:
                pos_reward = 1
                sum_rewards += 1
            else:
                pos_reward = 0

            positional_rewards.append(pos_reward)

        positional_rewards = np.array(positional_rewards)

        return positional_rewards
