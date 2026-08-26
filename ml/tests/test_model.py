import pytest

torch = pytest.importorskip("torch")

from tantra_ml.asr.model import AsrConfig, TantraAsr
from tantra_ml.tts.ctc_alignment import ctc_viterbi_alignment, token_durations_from_path


def test_small_asr_forward():
    config = AsrConfig(
        vocab_size=32, model_dim=64, layers=2, heads=4, feed_forward_dim=128,
        conv_channels=(16, 32, 48, 64), conv_kernels=(10, 8, 4, 4), conv_strides=(5, 4, 2, 2),
    )
    model = TantraAsr(config).eval()
    logits, lengths = model(torch.randn(2, 16_000), torch.tensor([16_000, 14_000]), torch.tensor([1, 10]))
    assert logits.shape[0] == 2
    assert logits.shape[-1] == 32
    assert torch.all(lengths > 0)


def test_ctc_alignment():
    logits = torch.full((7, 4), -8.0)
    path = [0, 1, 1, 0, 2, 2, 0]
    for frame, token in enumerate(path):
        logits[frame, token] = 0.0
    state_path = ctc_viterbi_alignment(logits.log_softmax(-1), torch.tensor([1, 2]))
    durations = token_durations_from_path(state_path, 2)
    assert durations.tolist() == [2, 2]
