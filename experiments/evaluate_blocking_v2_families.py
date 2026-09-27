"""Measure recall and candidate volume for individual V2 key families."""

from pathlib import Path
import argparse
import gc
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.candidate_generation.blocking import (  # noqa: E402
    build_address_numeric_index,
    build_address_token_index,
    build_name_prefix_index,
    build_name_token_index,
    generate_candidates_v2,
)
from src.candidate_generation.recall import parse_ground_truth_ids  # noqa: E402
from experiments.analyze_blocking_misses import (  # noqa: E402
    reservoir_sample_ground_truth,
    retrieve_records,
)


DATASET_DIR = ROOT / "data" / "student_resource" / "dataset" / "train"
FAMILY_ORDER = ["name_tokens", "name_prefix", "address_tokens", "address_numeric"]
FAMILY_LABELS = {
    "name_tokens": "country + sorted name tokens",
    "name_prefix": "country + significant-name prefix",
    "address_tokens": "sorted address tokens",
    "address_numeric": "country + numeric address components",
}


def parse_args():
    """Parse fixed-sample V2 family evaluation options."""
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
    """Update recovery counts for one Source 1 record."""
    recovered = true_ids & candidates
    stats["true_ids"] += len(true_ids)
    stats["recovered_ids"] += len(recovered)
    stats["all_recovered"] += true_ids <= candidates
    stats["any_recovered"] += bool(recovered)
    stats["zero_recovered"] += not recovered


def family_candidates(source_record, indexes, family, prefix_length):
    """Generate candidates from exactly one V2 family."""
    family_indexes = {name: indexes.get(name, {}) for name in FAMILY_ORDER}
    for name in FAMILY_ORDER:
        if name != family:
            family_indexes[name] = {}
    return generate_candidates_v2(source_record, family_indexes, prefix_length)


def build_one_family_index(path, family, frequency_cap, prefix_length):
    """Build one family index so large maps are not retained together."""
    builders = {
        "name_tokens": lambda: build_name_token_index(path, frequency_cap),
        "name_prefix": lambda: build_name_prefix_index(path, prefix_length, frequency_cap),
        "address_tokens": lambda: build_address_token_index(path, frequency_cap),
        "address_numeric": lambda: build_address_numeric_index(path, frequency_cap),
    }
    return builders[family]()


def metrics_for_candidates(sampled_truth, candidates_s2, candidates_s3):
    """Calculate metrics from bounded per-sample candidate sets."""
    stats = {
        "records": len(sampled_truth),
        "candidate_pairs": 0,
        "maximum_candidates": 0,
        "combined": empty_stats(),
        "s2": empty_stats(),
        "s3": empty_stats(),
    }
    for truth_record in sampled_truth:
        source_id = truth_record["source1_entity_id"]
        true_ids = parse_ground_truth_ids(truth_record["matched_entity_ids"])
        true_s2 = {value for value in true_ids if value.startswith("S2-")}
        true_s3 = {value for value in true_ids if value.startswith("S3-")}
        current_s2 = candidates_s2[source_id]
        current_s3 = candidates_s3[source_id]
        current = current_s2 | current_s3
        stats["candidate_pairs"] += len(current)
        stats["maximum_candidates"] = max(stats["maximum_candidates"], len(current))
        update_stats(stats["combined"], true_ids, current)
        update_stats(stats["s2"], true_s2, current_s2)
        update_stats(stats["s3"], true_s3, current_s3)
    return stats


def evaluate(sampled_truth, source_records, dataset_dir, frequency_cap, prefix_length):
    """Calculate isolated and incremental metrics one family at a time."""
    family_candidates_s2 = {source_id: set() for source_id in source_records}
    family_candidates_s3 = {source_id: set() for source_id in source_records}
    cumulative_s2 = {source_id: set() for source_id in source_records}
    cumulative_s3 = {source_id: set() for source_id in source_records}
    results = {}
    incremental_labels = ["key 1", "key 1 + key 2", "key 1 + key 2 + key 3", "all four keys"]

    for position, family in enumerate(FAMILY_ORDER):
        print(f"Building {family} index for Source 2...")
        index_s2 = build_one_family_index(
            dataset_dir / "train_source2.tsv", family, frequency_cap, prefix_length
        )
        for source_id, source_record in source_records.items():
            family_candidates_s2[source_id] = family_candidates(
                source_record, {family: index_s2}, family, prefix_length
            )
            cumulative_s2[source_id].update(family_candidates_s2[source_id])
        del index_s2
        gc.collect()

        print(f"Building {family} index for Source 3...")
        index_s3 = build_one_family_index(
            dataset_dir / "train_source3.tsv", family, frequency_cap, prefix_length
        )
        for source_id, source_record in source_records.items():
            family_candidates_s3[source_id] = family_candidates(
                source_record, {family: index_s3}, family, prefix_length
            )
            cumulative_s3[source_id].update(family_candidates_s3[source_id])
        del index_s3
        gc.collect()

        results[FAMILY_LABELS[family]] = metrics_for_candidates(
            sampled_truth, family_candidates_s2, family_candidates_s3
        )
        results[incremental_labels[position]] = metrics_for_candidates(
            sampled_truth, cumulative_s2, cumulative_s3
        )
    return results


def recall(stats):
    return stats["recovered_ids"] / stats["true_ids"] if stats["true_ids"] else 0.0


def print_stats(label, stats):
    """Print one family or incremental evaluation result."""
    print(f"\n{label}")
    print("-" * len(label))
    print(f"Candidate pairs: {stats['candidate_pairs']:,}")
    print(f"Average candidates/S1: {stats['candidate_pairs'] / stats['records']:.2f}")
    print(f"Maximum candidates/S1: {stats['maximum_candidates']:,}")
    for source in ("combined", "s2", "s3"):
        values = stats[source]
        print(
            f"{source.upper()} recovered: {values['recovered_ids']:,}/"
            f"{values['true_ids']:,}; recall: {recall(values):.2%}"
        )
    values = stats["combined"]
    print(f"Combined recall: {recall(values):.2%}")
    print(f"At least one true match recovered: {values['any_recovered']:,}")
    print(f"All true matches recovered: {values['all_recovered']:,}")
    print(f"Zero true matches recovered: {values['zero_recovered']:,}")


def run(sample_size, frequency_cap, prefix_length):
    """Run family attribution on the deterministic sample only."""
    sampled_truth, _ = reservoir_sample_ground_truth(
        DATASET_DIR / "train_ground_truth.tsv", sample_size
    )
    source_ids = [record["source1_entity_id"] for record in sampled_truth]
    source_records = retrieve_records(DATASET_DIR / "train_source1.tsv", source_ids)
    print(f"Sampled Source 1 records: {len(source_ids):,}")

    results = evaluate(
        sampled_truth, source_records, DATASET_DIR, frequency_cap, prefix_length
    )
    print("\nV2 key-family attribution")
    print("=" * 28)
    for label, stats in results.items():
        print_stats(label, stats)
    print("\nConfiguration: frequency cap=100, prefix length=8")
    print("V1 baseline: 28.80% combined recall; 22,430,431 full-dataset candidate pairs")
    return results


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments.sample_size, arguments.frequency_cap, arguments.prefix_length)
