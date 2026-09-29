import torch

from opsmind.ml.neural.dataset import IncidentTextDataset
from opsmind.ml.neural.vocabulary import Vocabulary


def test_neural_dataset_returns_expected_shapes() -> None:
    vocabulary = Vocabulary.build(
        [
            "database timeout",
            "network error",
        ]
    )

    dataset = IncidentTextDataset(
        texts=[
            "database timeout",
        ],
        labels=[
            1,
        ],
        vocabulary=vocabulary,
        max_length=5,
    )

    input_ids, attention_mask, label = dataset[0]

    assert input_ids.shape == torch.Size([5])
    assert attention_mask.shape == torch.Size([5])
    assert label.shape == torch.Size([])
    assert input_ids.dtype == torch.long
    assert attention_mask.dtype == torch.float32
    assert label.dtype == torch.long
