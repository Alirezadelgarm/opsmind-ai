from opsmind.ml.neural.vocabulary import Vocabulary


def test_vocabulary_encodes_padding_and_unknowns() -> None:
    vocabulary = Vocabulary.build(
        [
            "database timeout",
            "network timeout",
        ]
    )

    token_ids, attention_mask = vocabulary.encode(
        "database unseen",
        max_length=4,
    )

    assert token_ids[0] == vocabulary.token_to_id["database"]
    assert token_ids[1] == vocabulary.unk_id
    assert token_ids[2:] == [
        vocabulary.pad_id,
        vocabulary.pad_id,
    ]
    assert attention_mask == [
        1,
        1,
        0,
        0,
    ]
