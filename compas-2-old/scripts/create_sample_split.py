from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


DEFAULT_SOURCE = Path("data/compas-2x_pastries_features.csv")
DEFAULT_SAMPLE_SIZE = 10000
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42


def reservoir_sample_csv(source: Path, sample_size: int, seed: int) -> tuple[list[str], list[dict[str, str]]]:
    rng = random.Random(seed)
    sample: list[dict[str, str]] = []

    with source.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{source} has no header")

        for index, row in enumerate(reader):
            if index < sample_size:
                sample.append(row)
                continue

            replacement = rng.randint(0, index)
            if replacement < sample_size:
                sample[replacement] = row

    if len(sample) < sample_size:
        raise ValueError(f"requested {sample_size} rows, but {source} only contains {len(sample)}")

    return list(reader.fieldnames), sample


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a random sample and 80/20 train/test split.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    args = parser.parse_args()

    source = args.source
    stem = source.stem
    sample_path = source.with_name(f"{stem}_sample.csv")
    train_path = source.with_name(f"{stem}_sample_train.csv")
    test_path = source.with_name(f"{stem}_sample_test.csv")

    fieldnames, sample_rows = reservoir_sample_csv(source, args.sample_size, args.random_state)
    rng = random.Random(args.random_state)
    rng.shuffle(sample_rows)

    test_count = int(round(args.sample_size * args.test_size))
    test_rows = sample_rows[:test_count]
    train_rows = sample_rows[test_count:]

    write_csv(sample_path, fieldnames, sample_rows)
    write_csv(train_path, fieldnames, train_rows)
    write_csv(test_path, fieldnames, test_rows)

    print(f"Wrote {len(sample_rows)} sampled rows to {sample_path}")
    print(f"Wrote {len(train_rows)} train rows to {train_path}")
    print(f"Wrote {len(test_rows)} test rows to {test_path}")


if __name__ == "__main__":
    main()
