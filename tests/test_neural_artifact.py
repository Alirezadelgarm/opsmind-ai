from pathlib import Path

from opsmind.ml.neural.inference import NeuralIncidentClassifier
from opsmind.ml.neural.model import NeuralTextClassifier
from opsmind.ml.neural.training import _save_neural_artifact
from opsmind.ml.neural.vocabulary import Vocabulary


def test_neural_artifact_round_trip(
    tmp_path: Path,
) -> None:
    vocabulary = Vocabulary.build(
        [
            "database timeout",
            "network error",
            "application crash",
        ]
    )

    model = NeuralTextClassifier(
        vocab_size=len(vocabulary),
        embedding_dim=8,
        hidden_dim=6,
        num_classes=3,
        padding_idx=vocabulary.pad_id,
        dropout=0.0,
    )

    artifact_path = (
        tmp_path
        / "neural.pt"
    )

    _save_neural_artifact(
        model=model,
        vocabulary=vocabulary,
        classes=(
            "application",
            "database",
            "network",
        ),
        model_path=artifact_path,
        embedding_dim=8,
        hidden_dim=6,
        max_length=10,
        dropout=0.0,
    )

    loaded = NeuralIncidentClassifier.load(
        artifact_path
    )

    assert loaded is not None
