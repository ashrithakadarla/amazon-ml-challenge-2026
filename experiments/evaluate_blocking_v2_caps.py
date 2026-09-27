"""Measure V2 blocking sensitivity to frequency caps."""

from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.candidate_generation.blocking import build_v2_indexes  # noqa: E402
from experiments.analyze_blocking_misses import (  # noqa: E402
    reservoir_sample_ground_truth,
    retrieve_records,
)
from experiments.evaluate_blocking_v2 import (  # noqa: E402
    evaluate_sample,
    recall,
)


DATASET_DIR = ROOT / "data" / "student_resource" / "dataset" / "train"
CAPS = (25, 50, 75, 100, 150)
PREFIX_LENGTH = 8
MAX_CAP = max(CAPS)


def parse_args():
    """Parse fixed-sample cap sensitivity options."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=10_000)
    return parser.parse_args()


def cap_view(indexes, frequency_cap):
    """Return a lower-cap view without rebuilding the target indexes."""
    return {
        family: {
            key: entity_ids
            for key, entity_ids in family_index.items()
            if len(entity_ids) <= frequency_cap
        }
        for family, family_index in indexes.items()
    }


def print_result(frequency_cap, stats):
    """Print one cap's requested candidate and recall metrics."""
    combined = stats["combined"]
    print(f"\nFrequency cap: {frequency_cap}")
    print("-" * 24)
    print(f"Candidate pairs: {stats['candidate_pairs']:,}")
    print(f"Average candidates/S1: {stats['candidate_pairs'] / stats['records']:.2f}")
    print(f"Maximum candidates/S1: {stats['maximum_candidates']:,}")
    print(f"Combined recall: {recall(combined):.2%}")
    print(f"S2 recovered: {stats['s2']['recovered_ids']:,}; recall: {recall(stats['s2']):.2%}")
    print(f"S3 recovered: {stats['s3']['recovered_ids']:,}; recall: {recall(stats['s3']):.2%}")
    print(f"At least one recovered: {combined['any_recovered']:,}")
    print(f"All matches recovered: {combined['all_recovered']:,}")
    print(f"Zero recovered: {combined['zero_recovered']:,}")


def run(sample_size):
    """Evaluate all caps on the same deterministic sample."""
    sampled_truth, _ = reservoir_sample_ground_truth(
        DATASET_DIR / "train_ground_truth.tsv", sample_size
    )
    source_ids = [record["source1_entity_id"] for record in sampled_truth]
    source_records = retrieve_records(DATASET_DIR / "train_source1.tsv", source_ids)
    print(f"Sampled Source 1 records: {len(source_ids):,}")
    print(f"Building Source 2 V2 indexes once at cap {MAX_CAP}...")
    indexes_s2 = build_v2_indexes(
        DATASET_DIR / "train_source2.tsv",
        frequency_cap=MAX_CAP,
        prefix_length=PREFIX_LENGTH,
    )
    print(f"Building Source 3 V2 indexes once at cap {MAX_CAP}...")
    indexes_s3 = build_v2_indexes(
        DATASET_DIR / "train_source3.tsv",
        frequency_cap=MAX_CAP,
        prefix_length=PREFIX_LENGTH,
    )
    print("Indexes built. Evaluating caps...")
    results = {}
    for frequency_cap in CAPS:
        results[frequency_cap] = evaluate_sample(
            sampled_truth,
            source_records,
            cap_view(indexes_s2, frequency_cap),
            cap_view(indexes_s3, frequency_cap),
            PREFIX_LENGTH,
        )
        print_result(frequency_cap, results[frequency_cap])
    print("\nConfiguration: sample size=10,000, prefix length=8")
    print("V1 baseline: 28.80% combined recall; 22,430,431 full-dataset candidate pairs")
    return results


if __name__ == "__main__":
    run(parse_args().sample_size)
