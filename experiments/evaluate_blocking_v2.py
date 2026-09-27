"""Evaluate V2 blocking on the deterministic diagnostic sample."""

from collections import Counter
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.candidate_generation.blocking import (  # noqa: E402
    CHUNK_SIZE,
    build_v2_indexes,
    generate_candidates_v2,
)
from src.candidate_generation.recall import (  # noqa: E402
    iter_tsv_chunks,
    parse_ground_truth_ids,
)
from experiments.analyze_blocking_misses import (  # noqa: E402
    reservoir_sample_ground_truth,
    retrieve_records,
)


DATASET_DIR = ROOT / "data" / "student_resource" / "dataset" / "train"


def parse_args():
    """Parse bounded V2 evaluation options."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=10_000)
    parser.add_argument("--frequency-cap", type=int, default=100)
    parser.add_argument("--prefix-length", type=int, default=8)
    return parser.parse_args()


def empty_stats():
    return {
        "true_ids": 0,
        "recovered_ids": 0,
        "all_recovered": 0,
        "any_recovered": 0,
        "zero_recovered": 0,
    }


def update_stats(stats, true_ids, candidates):
    """Update recall counts for one source's true IDs and candidates."""
    recovered = true_ids & candidates
    stats["true_ids"] += len(true_ids)
    stats["recovered_ids"] += len(recovered)
    if true_ids <= candidates:
        stats["all_recovered"] += 1
    if recovered:
        stats["any_recovered"] += 1
    else:
        stats["zero_recovered"] += 1


def evaluate_sample(sampled_truth, source_records, indexes_s2, indexes_s3, prefix_length):
    """Generate V2 candidates and calculate sample metrics."""
    stats = {
        "records": 0,
        "candidate_pairs": 0,
        "maximum_candidates": 0,
        "combined": empty_stats(),
        "s2": empty_stats(),
        "s3": empty_stats(),
    }
    for truth_record in sampled_truth:
        source_record = source_records[truth_record["source1_entity_id"]]
        true_ids = parse_ground_truth_ids(truth_record["matched_entity_ids"])
        true_ids_s2 = {entity_id for entity_id in true_ids if entity_id.startswith("S2-")}
        true_ids_s3 = {entity_id for entity_id in true_ids if entity_id.startswith("S3-")}
        candidates_s2 = generate_candidates_v2(source_record, indexes_s2, prefix_length)
        candidates_s3 = generate_candidates_v2(source_record, indexes_s3, prefix_length)
        candidates = candidates_s2 | candidates_s3

        stats["records"] += 1
        stats["candidate_pairs"] += len(candidates)
        stats["maximum_candidates"] = max(stats["maximum_candidates"], len(candidates))
        update_stats(stats["combined"], true_ids, candidates)
        update_stats(stats["s2"], true_ids_s2, candidates_s2)
        update_stats(stats["s3"], true_ids_s3, candidates_s3)
    return stats


def recall(stats):
    return stats["recovered_ids"] / stats["true_ids"] if stats["true_ids"] else 0.0


def print_stats(stats):
    """Print the V2 sample metrics in a comparison-friendly format."""
    print("\nV2 blocking sample results")
    print("=" * 32)
    print(f"Sample Source 1 records: {stats['records']:,}")
    print(f"Candidate pairs: {stats['candidate_pairs']:,}")
    print(
        "Average candidates/S1: "
        f"{stats['candidate_pairs'] / stats['records']:.2f}"
    )
    print(f"Maximum candidates/S1: {stats['maximum_candidates']:,}")
    for label in ("combined", "s2", "s3"):
        values = stats[label]
        print(f"{label.upper()} recall: {recall(values):.2%}")
        print(f"  True IDs recovered: {values['recovered_ids']:,}/{values['true_ids']:,}")
        print(f"  All true matches recovered: {values['all_recovered']:,}")
        print(f"  At least one recovered: {values['any_recovered']:,}")
        print(f"  Zero recovered: {values['zero_recovered']:,}")


def run(sample_size, frequency_cap, prefix_length):
    """Run V2 evaluation without scanning all Source 1 records."""
    sampled_truth, _ = reservoir_sample_ground_truth(
        DATASET_DIR / "train_ground_truth.tsv", sample_size
    )
    source_ids = [record["source1_entity_id"] for record in sampled_truth]
    source_records = retrieve_records(DATASET_DIR / "train_source1.tsv", source_ids)
    print(f"Sampled {len(source_ids):,} Source 1 records.")

    print("Building capped V2 indexes for Source 2...")
    indexes_s2 = build_v2_indexes(
        DATASET_DIR / "train_source2.tsv",
        frequency_cap=frequency_cap,
        prefix_length=prefix_length,
    )
    print("Building capped V2 indexes for Source 3...")
    indexes_s3 = build_v2_indexes(
        DATASET_DIR / "train_source3.tsv",
        frequency_cap=frequency_cap,
        prefix_length=prefix_length,
    )
    print("V2 indexes built.")
    stats = evaluate_sample(
        sampled_truth, source_records, indexes_s2, indexes_s3, prefix_length
    )
    print_stats(stats)
    print("\nComparison baseline:")
    print("V1 combined recall: 28.80%")
    print("V1 full-dataset candidate pairs: 22,430,431")
    return stats


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments.sample_size, arguments.frequency_cap, arguments.prefix_length)
