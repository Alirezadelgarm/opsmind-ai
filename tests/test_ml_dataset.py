from pathlib import Path

from opsmind.ml.dataset import load_training_dataset


def test_load_training_dataset_cleans_and_deduplicates(
    tmp_path: Path,
) -> None:
    rows = [
        "text,category",
        "database timeout,Database",
        "database timeout,Database",
        "packet loss,Network",
        "dns failed,Network",
        "api crashed,Application",
        "worker crashed,Application",
        "sql deadlock,Database",
        "route missing,Network",
        "api error,Application",
        "slow query,Database",
        "firewall block,Network",
        "runtime error,Application",
    ]

    path = tmp_path / "incidents.csv"
    path.write_text(
        "\n".join(rows),
        encoding="utf-8",
    )

    frame = load_training_dataset(path)

    assert len(frame) == 11
    assert set(frame["category"]) == {
        "application",
        "database",
        "network",
    }
