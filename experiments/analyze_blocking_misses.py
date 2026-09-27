"""Sample and diagnose true matches missed by exact blocking."""

from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
import argparse
import random
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.candidate_generation.blocking import (  # noqa: E402
    CHUNK_SIZE,
    generate_candidates,
)
from src.candidate_generation.recall import (  # noqa: E402
    iter_tsv_chunks,
    parse_ground_truth_ids,
)
from src.preprocessing.normalize import (  # noqa: E402
    normalize_address,
    normalize_country,
    normalize_name,
)


DATASET_DIR = ROOT / "data" / "student_resource" / "dataset" / "train"
OUTPUT_DIR = ROOT / "experiments" / "outputs"
SAMPLE_PATH = OUTPUT_DIR / "blocking_misses_sample.tsv"
SUMMARY_PATH = OUTPUT_DIR / "blocking_misses_summary.txt"
RANDOM_SEED = 20260926
SAMPLE_PER_SOURCE = 100
SIMILARITY_THRESHOLD = 0.9

SAMPLE_COLUMNS = [
    "source1_entity_id",
    "source1_business_name",
    "source1_business_address",
    "source1_country",
    "matched_entity_id",
    "matched_source",
    "matched_business_name",
    "matched_business_address",
    "matched_country",
    "normalized_s1_name",
    "normalized_matched_name",
    "normalized_s1_address",
    "normalized_matched_address",
    "normalized_names_equal",
    "normalized_addresses_equal",
    "countries_equal",
    "name_first_token_matches",
    "name_first_5_chars_match",
    "address_token_overlap",
]


def parse_args():
    """Parse command-line options for the bounded diagnostic."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=10_000)
    return parser.parse_args()


def reservoir_sample_ground_truth(path, sample_size):
    """Select a deterministic reservoir sample of ground-truth records."""
    if sample_size <= 0:
        raise ValueError("sample-size must be positive")

    random_generator = random.Random(RANDOM_SEED)
    sample = []
    seen = 0
    for chunk in iter_tsv_chunks(path, CHUNK_SIZE):
        for record in chunk.to_dict("records"):
            seen += 1
            if len(sample) < sample_size:
                sample.append(record)
            else:
                replacement = random_generator.randrange(seen)
                if replacement < sample_size:
                    sample[replacement] = record
    sample.sort(key=lambda record: record["source1_entity_id"])
    return sample, seen


def retrieve_records(path, entity_ids):
    """Retrieve selected records from a large TSV in bounded chunks."""
    records = {}
    remaining = set(entity_ids)
    for chunk in iter_tsv_chunks(path, CHUNK_SIZE):
        selected = chunk[chunk["entity_id"].isin(remaining)]
        for record in selected.to_dict("records"):
            records[record["entity_id"]] = record
        remaining.difference_update(selected["entity_id"])
        if not remaining:
            break
    missing = set(entity_ids) - records.keys()
    if missing:
        raise ValueError(f"Missing {len(missing):,} requested records from {path.name}")
    return records


def add_index_value(index, key, entity_id):
    """Add an entity ID to an exact blocking key."""
    if key:
        index[key].add(entity_id)


def build_sample_indexes(records):
    """Build the current exact name/address indexes for selected targets only."""
    name_index = defaultdict(set)
    address_index = defaultdict(set)
    for record in records.values():
        entity_id = record["entity_id"]
        country = normalize_country(record.get("country"))
        name = normalize_name(record.get("business_name"))
        address = normalize_address(record.get("business_address"))
        if name:
            add_index_value(name_index, (country, name), entity_id)
            add_index_value(name_index, name, entity_id)
        if address:
            add_index_value(address_index, (country, address), entity_id)
    return dict(name_index), dict(address_index)


def first_token(value):
    return value.split(" ", 1)[0] if value else ""


def token_overlap(left, right):
    return bool(set(left.split()) & set(right.split())) if left and right else False


def slightly_differs(left, right):
    if not left or not right or left == right:
        return False
    return SequenceMatcher(None, left, right).ratio() >= SIMILARITY_THRESHOLD


def make_diagnostic(source_record, target_record):
    """Create requested comparison fields plus internal aggregate flags."""
    source_name = normalize_name(source_record.get("business_name"))
    target_name = normalize_name(target_record.get("business_name"))
    source_address = normalize_address(source_record.get("business_address"))
    target_address = normalize_address(target_record.get("business_address"))
    source_country = normalize_country(source_record.get("country"))
    target_country = normalize_country(target_record.get("country"))
    return {
        "source1_entity_id": source_record["entity_id"],
        "source1_business_name": source_record.get("business_name", ""),
        "source1_business_address": source_record.get("business_address", ""),
        "source1_country": source_record.get("country", ""),
        "matched_entity_id": target_record["entity_id"],
        "matched_source": target_record["_source"],
        "matched_business_name": target_record.get("business_name", ""),
        "matched_business_address": target_record.get("business_address", ""),
        "matched_country": target_record.get("country", ""),
        "normalized_s1_name": source_name,
        "normalized_matched_name": target_name,
        "normalized_s1_address": source_address,
        "normalized_matched_address": target_address,
        "normalized_names_equal": bool(source_name and source_name == target_name),
        "normalized_addresses_equal": bool(
            source_address and source_address == target_address
        ),
        "countries_equal": bool(source_country and source_country == target_country),
        "name_first_token_matches": bool(
            first_token(source_name) and first_token(source_name) == first_token(target_name)
        ),
        "name_first_5_chars_match": bool(
            source_name and target_name and source_name[:5] == target_name[:5]
        ),
        "address_token_overlap": token_overlap(source_address, target_address),
        "_name_slight": slightly_differs(source_name, target_name),
        "_address_slight": slightly_differs(source_address, target_address),
        "_both_differ": bool(
            source_name and target_name and source_name != target_name
            and source_address and target_address and source_address != target_address
        ),
        "_missing_address": not target_address,
        "_missing_fields": any(
            not value
            for value in (
                source_record.get("business_name", ""),
                source_record.get("business_address", ""),
                source_record.get("country", ""),
                target_record.get("business_name", ""),
                target_record.get("business_address", ""),
                target_record.get("country", ""),
            )
        ),
    }


def update_counts(counts, diagnostic):
    """Update aggregate categories for one missed true match."""
    if diagnostic["normalized_names_equal"]:
        counts["exact name match but blocked anyway"] += 1
    if diagnostic["normalized_addresses_equal"]:
        counts["exact address match but blocked anyway"] += 1
    if diagnostic["_name_slight"]:
        counts["name differs only slightly"] += 1
    if diagnostic["_address_slight"]:
        counts["address differs only slightly"] += 1
    if diagnostic["_both_differ"]:
        counts["both name and address differ"] += 1
    if diagnostic["_missing_address"]:
        counts["missing matched address"] += 1
    if diagnostic["_missing_fields"]:
        counts["missing source1/matched fields"] += 1


def write_outputs(sample_rows, counts, totals, sample_size, ground_truth_rows):
    """Write the sampled missed pairs and aggregate summary."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(sample_rows, columns=SAMPLE_COLUMNS).to_csv(
        SAMPLE_PATH, sep="\t", index=False
    )
    with SUMMARY_PATH.open("w", encoding="utf-8") as output:
        output.write("Blocking miss diagnostic summary\n")
        output.write("=" * 40 + "\n")
        output.write(f"Ground-truth rows scanned: {ground_truth_rows:,}\n")
        output.write(f"Source 1 records sampled: {sample_size:,}\n")
        output.write(f"Missed S2 true matches: {totals['S2']:,}\n")
        output.write(f"Missed S3 true matches: {totals['S3']:,}\n")
        output.write(f"Missed true matches total: {sum(totals.values()):,}\n")
        output.write(f"Sample rows written: {len(sample_rows):,}\n")
        output.write("\nAggregate diagnostic counts\n")
        for label, count in counts.items():
            output.write(f"{label}: {count:,}\n")
        output.write(f"\nSample output: {SAMPLE_PATH}\n")
        output.write(f"Summary output: {SUMMARY_PATH}\n")


def run(sample_size):
    """Run the bounded, deterministic blocking-miss diagnostic."""
    print(f"Selecting {sample_size:,} deterministic Source 1 records...")
    sampled_truth, ground_truth_rows = reservoir_sample_ground_truth(
        DATASET_DIR / "train_ground_truth.tsv", sample_size
    )
    source1_ids = [record["source1_entity_id"] for record in sampled_truth]
    print(f"Sampled {len(source1_ids):,} Source 1 records.")

    source1_records = retrieve_records(DATASET_DIR / "train_source1.tsv", source1_ids)
    matched_ids_by_source = {"S2": set(), "S3": set()}
    for truth_record in sampled_truth:
        for entity_id in parse_ground_truth_ids(truth_record["matched_entity_ids"]):
            source = "S2" if entity_id.startswith("S2-") else "S3"
            matched_ids_by_source[source].add(entity_id)

    target_records = {}
    for source in ("S2", "S3"):
        print(f"Retrieving {len(matched_ids_by_source[source]):,} {source} target records...")
        records = retrieve_records(
            DATASET_DIR / f"train_source{source[1:]}.tsv",
            matched_ids_by_source[source],
        )
        for record in records.values():
            record["_source"] = source
        target_records.update(records)

    name_index, address_index = build_sample_indexes(target_records)
    print("Indexes built for sampled target records.")

    truth_by_source1 = {
        record["source1_entity_id"]: parse_ground_truth_ids(record["matched_entity_ids"])
        for record in sampled_truth
    }
    sample_rows = []
    sample_counts = Counter()
    counts = Counter()
    totals = Counter()
    evaluated = 0
    for source1_id in source1_ids:
        source_record = source1_records[source1_id]
        candidates = generate_candidates(source_record, name_index, address_index)
        for matched_id in truth_by_source1[source1_id]:
            if matched_id in candidates:
                continue
            target_record = target_records[matched_id]
            diagnostic = make_diagnostic(source_record, target_record)
            source = target_record["_source"]
            totals[source] += 1
            update_counts(counts, diagnostic)
            if sample_counts[source] < SAMPLE_PER_SOURCE:
                sample_rows.append(
                    {key: diagnostic[key] for key in SAMPLE_COLUMNS}
                )
                sample_counts[source] += 1
        evaluated += 1
        if evaluated % 1_000 == 0 or evaluated == len(source1_ids):
            print(
                f"Evaluated {evaluated:,}/{len(source1_ids):,}; "
                f"missed S2: {totals['S2']:,}; missed S3: {totals['S3']:,}"
            )

    write_outputs(sample_rows, counts, totals, len(source1_ids), ground_truth_rows)
    print(f"Wrote sample: {SAMPLE_PATH}")
    print(f"Wrote summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    run(parse_args().sample_size)
