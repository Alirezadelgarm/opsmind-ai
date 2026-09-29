import torch

from opsmind.ml.neural.model import NeuralTextClassifier


def test_neural_model_forward_shape() -> None:
    model = NeuralTextClassifier(
        vocab_size=20,
        embedding_dim=8,
        hidden_dim=6,
        num_classes=3,
        padding_idx=0,
        dropout=0.0,
    )

    input_ids = torch.tensor(
        [
            [1, 2, 3, 0],
            [4, 5, 0, 0],
        ],
        dtype=torch.long,
    )

    attention_mask = torch.tensor(
        [
            [1, 1, 1, 0],
            [1, 1, 0, 0],
        ],
        dtype=torch.float32,
    )

    logits = model(
        input_ids,
        attention_mask,
    )

    assert logits.shape == torch.Size(
        [2, 3]
    )
