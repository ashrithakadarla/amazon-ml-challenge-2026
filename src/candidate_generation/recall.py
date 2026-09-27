"""Measure exact blocking recall against the entity-resolution ground truth."""

from pathlib import Path
import sys
import sqlite3
import tempfile

import pandas as pd

try:
	from .blocking import (
		CHUNK_SIZE,
		build_address_index,
		build_name_index,
		generate_candidates,
		iter_tsv_records,
	)
except ImportError:
	sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
	from src.candidate_generation.blocking import (  # type: ignore[no-redef]
		CHUNK_SIZE,
		build_address_index,
		build_name_index,
		generate_candidates,
		iter_tsv_records,
	)


DATASET_DIR = (
	Path(__file__).resolve().parents[2]
	/ "data"
	/ "student_resource"
	/ "dataset"
	/ "train"
)


def parse_ground_truth_ids(value):
	"""Parse comma-separated matched entity IDs while preserving each ID."""
	if not value:
		return set()
	return {entity_id for entity_id in str(value).split(",") if entity_id}


def _empty_source_stats():
	return {
		"true_ids": 0,
		"recovered_ids": 0,
		"all_recovered": 0,
		"any_recovered": 0,
		"zero_recovered": 0,
	}


def _update_source_stats(stats, true_ids, candidates):
	recovered = true_ids & candidates
	stats["true_ids"] += len(true_ids)
	stats["recovered_ids"] += len(recovered)
	if recovered:
		stats["any_recovered"] += 1
	else:
		stats["zero_recovered"] += 1
	if true_ids <= candidates:
		stats["all_recovered"] += 1


SQLITE_BATCH_SIZE = 900


def evaluate_recall(source1_chunks, ground_truth_connection, name_index_s2,
					address_index_s2, name_index_s3, address_index_s3):
	"""Evaluate candidate recall using chunk-sized SQLite truth lookups."""
	stats = {
		"source1_records": 0,
		"total_candidate_pairs": 0,
		"maximum_candidates": 0,
		"combined": _empty_source_stats(),
		"s2": _empty_source_stats(),
		"s3": _empty_source_stats(),
	}

	for source_chunk in source1_chunks:
		truth_values = fetch_ground_truth_values(
			ground_truth_connection, source_chunk["entity_id"].tolist()
		)
		for source_record in source_chunk.to_dict("records"):
			truth_value = truth_values[source_record["entity_id"]]
			true_ids = parse_ground_truth_ids(truth_value)

			candidates_s2 = generate_candidates(
				source_record, name_index_s2, address_index_s2
			)
			candidates_s3 = generate_candidates(
				source_record, name_index_s3, address_index_s3
			)
			candidates = candidates_s2 | candidates_s3
			true_ids_s2 = {entity_id for entity_id in true_ids if entity_id.startswith("S2-")}
			true_ids_s3 = {entity_id for entity_id in true_ids if entity_id.startswith("S3-")}

			stats["source1_records"] += 1
			stats["total_candidate_pairs"] += len(candidates)
			stats["maximum_candidates"] = max(stats["maximum_candidates"], len(candidates))
			_update_source_stats(stats["combined"], true_ids, candidates)
			_update_source_stats(stats["s2"], true_ids_s2, candidates_s2)
			_update_source_stats(stats["s3"], true_ids_s3, candidates_s3)

		print(f"Processed {stats['source1_records']:,} Source 1 records...")

	return stats


def build_ground_truth_table(path, chunksize=CHUNK_SIZE):
	"""Build a temporary SQLite truth table from chunked TSV input."""
	database = tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False)
	database.close()
	connection = sqlite3.connect(database.name)
	connection.execute(
		"CREATE TABLE ground_truth ("
		"source1_entity_id TEXT PRIMARY KEY, matched_entity_ids TEXT)"
	)
	for chunk in iter_tsv_chunks(path, chunksize):
		connection.executemany(
			"INSERT INTO ground_truth (source1_entity_id, matched_entity_ids) "
			"VALUES (?, ?)",
			chunk[["source1_entity_id", "matched_entity_ids"]].itertuples(
				index=False, name=None
			),
		)
		connection.commit()
	return connection, database.name


def fetch_ground_truth_values(connection, source1_ids, batch_size=SQLITE_BATCH_SIZE):
	"""Fetch one Source 1 chunk using parameterized batched SQL queries."""
	values = {}
	for start in range(0, len(source1_ids), batch_size):
		batch = source1_ids[start:start + batch_size]
		placeholders = ",".join("?" for _ in batch)
		rows = connection.execute(
			"SELECT source1_entity_id, matched_entity_ids FROM ground_truth "
			f"WHERE source1_entity_id IN ({placeholders})",
			batch,
		).fetchall()
		values.update(rows)
	missing_ids = set(source1_ids) - values.keys()
	if missing_ids:
		raise ValueError(
			f"Missing ground-truth rows for {len(missing_ids):,} Source 1 IDs"
		)
	return values


def iter_tsv_chunks(path, chunksize=CHUNK_SIZE):
	"""Yield pandas chunks for bounded, sequential file processing."""
	return pd.read_csv(
		path,
		sep="\t",
		dtype=str,
		keep_default_na=False,
		chunksize=chunksize,
	)


def _recall(stats):
	return (
		stats["recovered_ids"] / stats["true_ids"]
		if stats["true_ids"]
		else 0.0
	)


def print_summary(stats):
	"""Print a readable recall and candidate-volume summary."""
	print("\nCandidate-generation recall summary")
	print("=" * 40)
	print(f"Source 1 records: {stats['source1_records']:,}")
	print(f"Total generated candidate pairs: {stats['total_candidate_pairs']:,}")
	print(
		"Average candidates per Source 1 record: "
		f"{stats['total_candidate_pairs'] / stats['source1_records']:.2f}"
		if stats["source1_records"]
		else "Average candidates per Source 1 record: 0.00"
	)
	print(f"Maximum candidates for one record: {stats['maximum_candidates']:,}")

	for label in ("combined", "s2", "s3"):
		source_stats = stats[label]
		print(f"\n{label.upper()} recall:")
		print(
			f"  Recovered true IDs: {source_stats['recovered_ids']:,} / "
			f"{source_stats['true_ids']:,} ({_recall(source_stats):.2%})"
		)
		print(f"  Records with ALL true matches recovered: {source_stats['all_recovered']:,}")
		print(f"  Records with at least one match recovered: {source_stats['any_recovered']:,}")
		print(f"  Records with zero matches recovered: {source_stats['zero_recovered']:,}")


def run_full_evaluation(dataset_dir=DATASET_DIR):
	"""Build target indexes and evaluate Source 1 against ground truth."""
	print("Building Source 2 exact blocking indexes...")
	name_index_s2 = build_name_index(dataset_dir / "train_source2.tsv")
	address_index_s2 = build_address_index(dataset_dir / "train_source2.tsv")
	print("Building Source 3 exact blocking indexes...")
	name_index_s3 = build_name_index(dataset_dir / "train_source3.tsv")
	address_index_s3 = build_address_index(dataset_dir / "train_source3.tsv")

	print("Evaluating Source 1 in chunks...")
	connection, database_path = build_ground_truth_table(
		dataset_dir / "train_ground_truth.tsv", CHUNK_SIZE
	)
	try:
		stats = evaluate_recall(
			iter_tsv_chunks(dataset_dir / "train_source1.tsv", CHUNK_SIZE),
			connection,
			name_index_s2,
			address_index_s2,
			name_index_s3,
			address_index_s3,
		)
	finally:
		connection.close()
		Path(database_path).unlink(missing_ok=True)
	print_summary(stats)
	return stats


def _run_tiny_test():
	"""Verify multi-match, partial-recovery, and zero-recovery accounting."""
	target_s2 = [
		{"entity_id": "S2-1", "business_name": "Alpha", "business_address": "1 Main", "country": "US"},
		{"entity_id": "S2-2", "business_name": "Alpha", "business_address": "2 Main", "country": "US"},
	]
	target_s3 = [
		{"entity_id": "S3-1", "business_name": "Beta", "business_address": "3 Main", "country": "US"},
	]
	source = [
		{"entity_id": "S1-1", "business_name": "Alpha", "business_address": "1 Main", "country": "US"},
		{"entity_id": "S1-2", "business_name": "Missing", "business_address": "", "country": "US"},
	]
	truth = [
		{"source1_entity_id": "S1-1", "matched_entity_ids": "S2-1,S2-2,S3-1"},
		{"source1_entity_id": "S1-2", "matched_entity_ids": "S3-99"},
	]
	stats = evaluate_recall(
		[pd.DataFrame(source)],
		_tiny_truth_connection(truth),
		build_name_index(target_s2),
		build_address_index(target_s2),
		build_name_index(target_s3),
		build_address_index(target_s3),
	)
	assert stats["combined"]["true_ids"] == 4
	assert stats["combined"]["recovered_ids"] == 2
	assert stats["combined"]["all_recovered"] == 0
	assert stats["combined"]["any_recovered"] == 1
	assert stats["combined"]["zero_recovered"] == 1
	assert stats["total_candidate_pairs"] == 2
	print("Tiny recall test passed.")


def _tiny_truth_connection(records):
	"""Create an in-memory truth table for tiny evaluator tests."""
	connection = sqlite3.connect(":memory:")
	connection.execute(
		"CREATE TABLE ground_truth ("
		"source1_entity_id TEXT PRIMARY KEY, matched_entity_ids TEXT)"
	)
	connection.executemany(
		"INSERT INTO ground_truth VALUES (?, ?)",
		(
			(record["source1_entity_id"], record["matched_entity_ids"])
			for record in records
		),
	)
	return connection


def _run_reordered_join_test():
	"""Verify Source 1 and truth rows can be joined in different orders."""
	connection = _tiny_truth_connection([
		{"source1_entity_id": "S1-2", "matched_entity_ids": "S3-99"},
		{"source1_entity_id": "S1-1", "matched_entity_ids": "S2-1"},
	])
	try:
		values = fetch_ground_truth_values(connection, ["S1-1", "S1-2"])
		assert values["S1-1"] == "S2-1"
		assert values["S1-2"] == "S3-99"
	finally:
		connection.close()
	print("Reordered ground-truth join test passed.")


if __name__ == "__main__":
	_run_tiny_test()
	_run_reordered_join_test()
	run_full_evaluation()
