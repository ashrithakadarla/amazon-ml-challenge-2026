"""Sample-first baseline matching model using similarity features."""

from pathlib import Path
import argparse
import gc
import sys

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

from src.candidate_generation.blocking import (  # noqa: E402
	V2_FREQUENCY_CAP,
	V2_PREFIX_LENGTH,
	build_address_numeric_index,
	build_address_token_index,
	build_name_prefix_index,
	build_name_token_index,
	generate_candidates_v2,
)
from src.features.similarity_features import (  # noqa: E402
	build_similarity_features,
)


DATASET_DIR = ROOT / "data" / "student_resource" / "dataset" / "train"
FEATURE_COLUMNS = [
	"name_ratio",
	"name_token_set_ratio",
	"name_partial_ratio",
	"name_exact",
	"address_ratio",
	"address_token_set_ratio",
	"address_partial_ratio",
	"address_exact",
	"address_missing",
	"country_match",
]
THRESHOLDS = (0.50, 0.60, 0.70, 0.80, 0.90)


def _load_records_by_id(path, entity_ids, chunksize=200_000):
	"""Load only requested records from a TSV in bounded chunks."""
	remaining = set(entity_ids)
	records = {}
	for chunk in pd.read_csv(
		path,
		sep="\t",
		dtype=str,
		keep_default_na=False,
		chunksize=chunksize,
	):
		selected = chunk[chunk["entity_id"].isin(remaining)]
		for record in selected.to_dict("records"):
			records[record["entity_id"]] = record
		remaining.difference_update(selected["entity_id"])
		if not remaining:
			break
	if remaining:
		raise ValueError(f"Missing {len(remaining):,} records from {path.name}")
	return records


def load_training_sample(dataset_dir=DATASET_DIR, sample_size=1_000):
	"""Load a deterministic bounded ground-truth sample and its S1 records."""
	truth = pd.read_csv(
		dataset_dir / "train_ground_truth.tsv",
		sep="\t",
		dtype=str,
		keep_default_na=False,
		nrows=sample_size,
	)
	source1_records = _load_records_by_id(
		dataset_dir / "train_source1.tsv", truth["source1_entity_id"]
	)
	return truth, source1_records


def generate_training_candidates(
	truth,
	source1_records,
	dataset_dir=DATASET_DIR,
	frequency_cap=V2_FREQUENCY_CAP,
	prefix_length=V2_PREFIX_LENGTH,
):
	"""Generate one independently scored pair for every bounded V2 candidate."""
	candidates_by_source = {
		source1_id: set() for source1_id in truth["source1_entity_id"]
	}
	family_builders = {
		"name_tokens": lambda path: build_name_token_index(path, frequency_cap),
		"name_prefix": lambda path: build_name_prefix_index(
			path, prefix_length, frequency_cap
		),
		"address_tokens": lambda path: build_address_token_index(
			path, frequency_cap
		),
		"address_numeric": lambda path: build_address_numeric_index(
			path, frequency_cap
		),
	}
	for family, build_index in family_builders.items():
		for source_number in (2, 3):
			index = build_index(dataset_dir / f"train_source{source_number}.tsv")
			family_indexes = {
				"name_tokens": {},
				"name_prefix": {},
				"address_tokens": {},
				"address_numeric": {},
			}
			family_indexes[family] = index
			for source1_id, source1 in source1_records.items():
				candidates_by_source[source1_id].update(
					generate_candidates_v2(source1, family_indexes, prefix_length)
				)
			del index
			gc.collect()
	pairs = []
	for source1_id in truth["source1_entity_id"]:
		pairs.extend(
			(source1_id, candidate_id)
			for candidate_id in candidates_by_source[source1_id]
		)
	return pd.DataFrame(
		pairs, columns=["source1_entity_id", "candidate_entity_id"]
	)


def add_training_labels(candidate_pairs, truth):
	"""Label each candidate independently from comma-separated ground truth."""
	truth_by_source = truth.set_index("source1_entity_id")["matched_entity_ids"]
	true_matches = {
		source1_id: {
			entity_id
			for entity_id in str(value).split(",")
			if entity_id
		}
		for source1_id, value in truth_by_source.items()
	}
	labels = [
		int(candidate_id in true_matches.get(source1_id, set()))
		for source1_id, candidate_id in candidate_pairs[
			["source1_entity_id", "candidate_entity_id"]
		].itertuples(index=False, name=None)
	]
	return candidate_pairs.assign(label=labels)


def train_classifier(training_frame):
	"""Train a simple balanced logistic-regression pair classifier."""
	model = make_pipeline(
		StandardScaler(),
		LogisticRegression(max_iter=1_000, class_weight="balanced", random_state=42),
	)
	model.fit(training_frame[FEATURE_COLUMNS], training_frame["label"])
	return model


def evaluate_thresholds(model, validation_frame, thresholds=THRESHOLDS):
	"""Score every pair and calculate precision, recall, and F0.5 per threshold."""
	scored = validation_frame.copy()
	scored["probability"] = model.predict_proba(scored[FEATURE_COLUMNS])[:, 1]
	results = []
	for threshold in thresholds:
		predicted = scored["probability"] >= threshold
		results.append(
			{
				"threshold": threshold,
				"precision": precision_score(
					scored["label"], predicted, zero_division=0
				),
				"recall": recall_score(scored["label"], predicted, zero_division=0),
				"f0.5": fbeta_score(
					scored["label"], predicted, beta=0.5, zero_division=0
				),
			}
		)
	return pd.DataFrame(results), scored


def split_by_source1(candidate_frame, validation_size=0.25, random_state=42):
	"""Split whole S1 groups so candidates from one S1 never cross splits."""
	source1_ids = candidate_frame["source1_entity_id"].drop_duplicates()
	train_ids, validation_ids = train_test_split(
		source1_ids, test_size=validation_size, random_state=random_state
	)
	return (
		candidate_frame[candidate_frame["source1_entity_id"].isin(train_ids)].copy(),
		candidate_frame[
			candidate_frame["source1_entity_id"].isin(validation_ids)
		].copy(),
	)


def run_sample_training(
	dataset_dir=DATASET_DIR,
	sample_size=1_000,
	frequency_cap=V2_FREQUENCY_CAP,
	prefix_length=V2_PREFIX_LENGTH,
):
	"""Run bounded training/evaluation and return model, metrics, and scores."""
	truth, source1_records = load_training_sample(dataset_dir, sample_size)
	candidate_pairs = generate_training_candidates(
		truth,
		source1_records,
		dataset_dir,
		frequency_cap,
		prefix_length,
	)
	candidate_ids = set(candidate_pairs["candidate_entity_id"])
	source2_records = _load_records_by_id(
		dataset_dir / "train_source2.tsv",
		{entity_id for entity_id in candidate_ids if entity_id.startswith("S2-")},
	)
	source3_records = _load_records_by_id(
		dataset_dir / "train_source3.tsv",
		{entity_id for entity_id in candidate_ids if entity_id.startswith("S3-")},
	)
	features = build_similarity_features(
		candidate_pairs, source1_records, source2_records, source3_records
	)
	training_frame, validation_frame = split_by_source1(
		add_training_labels(features, truth)
	)
	model = train_classifier(training_frame)
	metrics, scored_validation = evaluate_thresholds(model, validation_frame)
	selected = metrics.sort_values(
		["f0.5", "threshold"], ascending=[False, False]
	).iloc[0].to_dict()
	return {
		"model": model,
		"metrics": metrics,
		"selected": selected,
		"scored_validation": scored_validation,
		"candidate_pairs": len(candidate_pairs),
		"training_pairs": len(training_frame),
		"validation_pairs": len(validation_frame),
	}


def _run_smoke_test():
	"""Verify independent pair scoring, including multiple and zero matches."""
	truth = pd.DataFrame(
		[
			{"source1_entity_id": "S1-1", "matched_entity_ids": "S2-1,S2-2"},
			{"source1_entity_id": "S1-2", "matched_entity_ids": ""},
		]
	)
	features = pd.DataFrame(
		[
			{"source1_entity_id": "S1-1", "candidate_entity_id": "S2-1", **dict.fromkeys(FEATURE_COLUMNS, 1.0)},
			{"source1_entity_id": "S1-1", "candidate_entity_id": "S2-2", **dict.fromkeys(FEATURE_COLUMNS, 0.9)},
			{"source1_entity_id": "S1-2", "candidate_entity_id": "S2-3", **dict.fromkeys(FEATURE_COLUMNS, 0.0)},
		]
	)
	labeled = add_training_labels(features, truth)
	assert labeled["label"].tolist() == [1, 1, 0]
	model = train_classifier(pd.concat([labeled, labeled.iloc[[2]]]))
	metrics, scored = evaluate_thresholds(model, labeled)
	assert len(metrics) == len(THRESHOLDS)
	assert len(scored) == 3
	print(metrics.to_string(index=False))


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--sample-size", type=int, default=1_000)
	parser.add_argument("--smoke-test", action="store_true")
	arguments = parser.parse_args()
	if arguments.smoke_test:
		_run_smoke_test()
		return
	result = run_sample_training(sample_size=arguments.sample_size)
	print(f"Candidate pairs: {result['candidate_pairs']:,}")
	print(f"Training pairs: {result['training_pairs']:,}")
	print(f"Validation pairs: {result['validation_pairs']:,}")
	print(result["metrics"].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
	print("Selected threshold:", result["selected"])


if __name__ == "__main__":
	main()
