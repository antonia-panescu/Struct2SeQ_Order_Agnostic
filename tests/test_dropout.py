"""Smoke test for the shared-mask dropout module (CPU only)."""
import torch

from dropout import Dropout


def test_dropout_identity_in_eval_mode():
    drop = Dropout(r=0.5, batch_dim=None)
    drop.eval()
    x = torch.randn(2, 4, 8)
    out = drop(x)
    assert out.shape == x.shape
    torch.testing.assert_close(out, x)


def test_dropout_zeros_some_entries_in_train_mode():
    torch.manual_seed(0)
    drop = Dropout(r=0.5, batch_dim=None)
    drop.train()
    x = torch.ones(64, 64)
    out = drop(x)
    assert out.shape == x.shape
    zero_frac = (out == 0).float().mean().item()
    assert 0.2 < zero_frac < 0.8


def test_shared_dropout_shares_mask_along_batch():
    torch.manual_seed(0)
    drop = Dropout(r=0.5, batch_dim=0)
    drop.train()
    x = torch.ones(8, 32, 16)
    out = drop(x)
    first = (out[0] == 0)
    for i in range(1, x.shape[0]):
        torch.testing.assert_close((out[i] == 0).to(torch.uint8), first.to(torch.uint8))
