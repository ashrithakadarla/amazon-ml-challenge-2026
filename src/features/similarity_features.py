"""Similarity features for Source 1 to Source 2/3 candidate pairs."""

from collections.abc import Iterable, Mapping
from pathlib import Path
import sys

import pandas as pd
from rapidfuzz import fuzz

try:
	from ..preprocessing.normalize import (
		normalize_address,
		normalize_country,
		normalize_name,
	)
except ImportError:
	sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
	from src.preprocessing.normalize import (  # type: ignore[no-redef]
		normalize_address,
		normalize_country,
		normalize_name,
	)


FEATURE_COLUMNS = [
	"source1_entity_id",
	"candidate_entity_id",
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


def _record_map(records):
	if isinstance(records, pd.DataFrame):
		return records.set_index("entity_id").to_dict("index")
	return records


def _pair_rows(candidate_pairs):
	if isinstance(candidate_pairs, pd.DataFrame):
		columns = {"source1_entity_id", "candidate_entity_id"}
		if not columns.issubset(candidate_pairs.columns):
			raise ValueError(
				"candidate_pairs DataFrame must contain source1_entity_id "
				"and candidate_entity_id columns"
			)
		return candidate_pairs[
			["source1_entity_id", "candidate_entity_id"]
		].itertuples(index=False, name=None)

	return candidate_pairs


def _similarity_score(scorer, left, right):
	if not left or not right:
		return 0.0
	return float(scorer(left, right))


def build_similarity_features(
	candidate_pairs: Iterable,
	source1_records: Mapping,
	source2_records: Mapping,
	source3_records: Mapping,
) -> pd.DataFrame:
	"""Build normalized similarity features for each blocked candidate pair.

	Args:
		candidate_pairs: Iterable of ``(source1_entity_id, candidate_entity_id)``
			pairs, or a DataFrame with those two named columns.
		source1_records: Mapping or DataFrame of Source 1 records keyed by
			``entity_id``.
		source2_records: Mapping or DataFrame of Source 2 records keyed by
			``entity_id``.
		source3_records: Mapping or DataFrame of Source 3 records keyed by
			``entity_id``.

	Returns:
		A pandas DataFrame with IDs and the ten requested numeric features.
		Empty values score zero; ``address_missing`` is one when either address
		is empty after normalization.
	"""
	source1_records = _record_map(source1_records)
	target_records = dict(_record_map(source2_records))
	target_records.update(_record_map(source3_records))
	rows = []

	for source1_id, candidate_id in _pair_rows(candidate_pairs):
		try:
			source1 = source1_records[source1_id]
			target = target_records[candidate_id]
		except KeyError as error:
			raise KeyError(f"Missing record for candidate pair: {error.args[0]}") from error

		name_left = normalize_name(source1.get("business_name"))
		name_right = normalize_name(target.get("business_name"))
		address_left = normalize_address(source1.get("business_address"))
		address_right = normalize_address(target.get("business_address"))
		country_left = normalize_country(source1.get("country"))
		country_right = normalize_country(target.get("country"))

		rows.append(
			{
				"source1_entity_id": source1_id,
				"candidate_entity_id": candidate_id,
				"name_ratio": _similarity_score(fuzz.ratio, name_left, name_right),
				"name_token_set_ratio": _similarity_score(
					fuzz.token_set_ratio, name_left, name_right
				),
				"name_partial_ratio": _similarity_score(
					fuzz.partial_ratio, name_left, name_right
				),
				"name_exact": int(bool(name_left) and name_left == name_right),
				"address_ratio": _similarity_score(
					fuzz.ratio, address_left, address_right
				),
				"address_token_set_ratio": _similarity_score(
					fuzz.token_set_ratio, address_left, address_right
				),
				"address_partial_ratio": _similarity_score(
					fuzz.partial_ratio, address_left, address_right
				),
				"address_exact": int(
					bool(address_left) and address_left == address_right
				),
				"address_missing": int(not address_left or not address_right),
				"country_match": int(
					bool(country_left) and country_left == country_right
				),
			}
		)

	return pd.DataFrame(rows, columns=FEATURE_COLUMNS)


if __name__ == "__main__":
	source1 = {
		"S1-1": {
			"entity_id": "S1-1",
			"business_name": "Cafe Société",
			"business_address": "12 Main St.",
			"country": "US",
		}
	}
	source2 = {
		"S2-1": {
			"entity_id": "S2-1",
			"business_name": "Cafe Societe",
			"business_address": "12 Main Street",
			"country": "US",
		}
	}
	source3 = {
		"S3-1": {
			"entity_id": "S3-1",
			"business_name": "Other Shop",
			"business_address": "",
			"country": "India",
		}
	}
	features = build_similarity_features(
		[("S1-1", "S2-1"), ("S1-1", "S3-1")], source1, source2, source3
	)
	assert len(features) == 2
	assert features.loc[0, "country_match"] == 1
	assert features.loc[1, "address_missing"] == 1
	assert features.loc[0, "name_exact"] == 0
	dataframe_features = build_similarity_features(
		pd.DataFrame(
			[
				{"source1_entity_id": "S1-1", "candidate_entity_id": "S2-1"}
			]
		),
		source1,
		source2,
		source3,
	)
	assert dataframe_features.loc[0, "candidate_entity_id"] == "S2-1"
	print(features.to_string(index=False))
