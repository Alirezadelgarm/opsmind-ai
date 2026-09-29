from pathlib import Path

from opsmind.ml.classifier import (
    IncidentClassifier,
    build_text_classifier,
)
from opsmind.models import (
    DocumentMetadata,
    ProcessedDocument,
)


def test_classifier_predicts_processed_document() -> None:
    pipeline = build_text_classifier(
        random_state=42
    )

    texts = [
        "postgresql deadlock database query",
        "mysql database connection timeout",
        "packet loss network routing",
        "dns network failure",
        "application crashed exception",
        "python application runtime error",
    ]
    labels = [
        "database",
        "database",
        "network",
        "network",
        "application",
        "application",
    ]

    pipeline.fit(texts, labels)

    classifier = IncidentClassifier(pipeline)

    document = ProcessedDocument(
        title="DNS outage",
        content="Packet loss and DNS resolution failed.",
        metadata=DocumentMetadata(
            source_path=Path("network.txt"),
            file_type="txt",
            size_bytes=30,
        ),
        word_count=6,
        character_count=38,
    )

    prediction = classifier.predict_documents(
        [document]
    )[0]

    assert prediction.predicted_category == "network"
    assert 0.0 <= prediction.confidence <= 1.0
    assert set(prediction.probabilities) == {
        "application",
        "database",
        "network",
    }
