"""Exact blocking indexes for entity-resolution candidate generation."""

from collections import defaultdict
import re
from pathlib import Path
import sys

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


CHUNK_SIZE = 200_000

# Locked production V2 configuration. All four families are unioned, and keys
# exceeding the frequency cap are omitted to keep candidate volume bounded.
V2_FREQUENCY_CAP = 75
V2_PREFIX_LENGTH = 8
V2_KEY_FAMILIES = (
	"name_tokens",
	"name_prefix",
	"address_tokens",
	"address_numeric",
)


def iter_tsv_records(path, chunksize=CHUNK_SIZE):
	"""Yield source records from a TSV in bounded pandas chunks."""
	import pandas as pd

	for chunk in pd.read_csv(
		path,
		sep="\t",
		dtype=str,
		keep_default_na=False,
		chunksize=chunksize,
	):
		yield from chunk.to_dict("records")


def _records(source):
	if isinstance(source, (str, Path)):
		return iter_tsv_records(source)
	return source


def _add(index, key, entity_id):
	if key:
		index[key].add(entity_id)


def build_name_index(records):
	"""Build country-aware and bare normalized-name to ID lookup sets."""
	index = defaultdict(set)
	for record in _records(records):
		entity_id = record["entity_id"]
		country = normalize_country(record.get("country"))
		name = normalize_name(record.get("business_name"))
		if not name:
			continue
		_add(index, (country, name), entity_id)
		_add(index, name, entity_id)
	return dict(index)


def build_address_index(records):
	"""Build a country-aware normalized-address to ID lookup set."""
	index = defaultdict(set)
	for record in _records(records):
		address = normalize_address(record.get("business_address"))
		if not address:
			continue
		country = normalize_country(record.get("country"))
		_add(index, (country, address), record["entity_id"])
	return dict(index)


def generate_candidates(source_record, name_index, address_index):
	"""Return exact-blocking candidate IDs for one Source 1 record."""
	country = normalize_country(source_record.get("country"))
	name = normalize_name(source_record.get("business_name"))
	address = normalize_address(source_record.get("business_address"))
	candidates = set()

	if name:
		candidates.update(name_index.get((country, name), ()))
		candidates.update(name_index.get(name, ()))
	if address:
		candidates.update(address_index.get((country, address), ()))
	return candidates


def _name_tokens(value):
	return tuple(sorted(set(normalize_name(value).split())))


def _significant_name_token(value):
	tokens = [token for token in normalize_name(value).split() if len(token) >= 3]
	return max(tokens, key=lambda token: (len(token), token), default="")


def _address_tokens(value):
	return tuple(sorted(set(normalize_address(value).split())))


def _address_numbers(value):
	return tuple(sorted(set(re.findall(r"\d+", normalize_address(value)))))


def _capped_add(index, blocked, key, entity_id, frequency_cap):
	if not key or key in blocked:
		return
	values = index.setdefault(key, set())
	values.add(entity_id)
	if len(values) > frequency_cap:
		index.pop(key, None)
		blocked.add(key)


def build_name_token_index(records, frequency_cap=V2_FREQUENCY_CAP):
	"""Index country-aware sorted normalized name tokens with a frequency cap."""
	index = defaultdict(set)
	blocked = set()
	for record in _records(records):
		country = normalize_country(record.get("country"))
		tokens = _name_tokens(record.get("business_name"))
		key = (country, tokens) if tokens else ()
		_capped_add(index, blocked, key, record["entity_id"], frequency_cap)
	return dict(index)


def build_name_prefix_index(
	records,
	prefix_length=V2_PREFIX_LENGTH,
	frequency_cap=V2_FREQUENCY_CAP,
):
	"""Index country-aware significant-name prefixes with a frequency cap."""
	index = defaultdict(set)
	blocked = set()
	for record in _records(records):
		country = normalize_country(record.get("country"))
		token = _significant_name_token(record.get("business_name"))
		key = (country, token[:prefix_length]) if token else ()
		_capped_add(index, blocked, key, record["entity_id"], frequency_cap)
	return dict(index)


def build_address_token_index(records, frequency_cap=V2_FREQUENCY_CAP):
	"""Index country-aware sorted address tokens with a frequency cap."""
	index = defaultdict(set)
	blocked = set()
	for record in _records(records):
		country = normalize_country(record.get("country"))
		tokens = _address_tokens(record.get("business_address"))
		key = (country, tokens) if tokens else ()
		_capped_add(index, blocked, key, record["entity_id"], frequency_cap)
	return dict(index)


def build_address_numeric_index(records, frequency_cap=V2_FREQUENCY_CAP):
	"""Index country-aware numeric address components with a frequency cap."""
	index = defaultdict(set)
	blocked = set()
	for record in _records(records):
		country = normalize_country(record.get("country"))
		key = (country, _address_numbers(record.get("business_address")))
		if key[1]:
			_capped_add(index, blocked, key, record["entity_id"], frequency_cap)
	return dict(index)


def build_v2_indexes(
	records,
	frequency_cap=V2_FREQUENCY_CAP,
	prefix_length=V2_PREFIX_LENGTH,
):
	"""Build the four capped production V2 blocking-family indexes."""
	indexes = {
		name: defaultdict(set) for name in V2_KEY_FAMILIES
	}
	blocked = {name: set() for name in indexes}
	for record in _records(records):
		entity_id = record["entity_id"]
		country = normalize_country(record.get("country"))
		name_tokens = _name_tokens(record.get("business_name"))
		name_token = _significant_name_token(record.get("business_name"))
		address_tokens = _address_tokens(record.get("business_address"))
		address_numbers = _address_numbers(record.get("business_address"))
		keys = {
			"name_tokens": (country, name_tokens) if name_tokens else (),
			"name_prefix": (country, name_token[:prefix_length]) if name_token else (),
			"address_tokens": (country, address_tokens),
			"address_numeric": (country, address_numbers) if address_numbers else (),
		}
		for name, key in keys.items():
			_capped_add(indexes[name], blocked[name], key, entity_id, frequency_cap)
	return {name: dict(index) for name, index in indexes.items()}


def generate_candidates_v2(
	source_record,
	indexes,
	prefix_length=V2_PREFIX_LENGTH,
):
	"""Union candidates from all four production V2 blocking families."""
	country = normalize_country(source_record.get("country"))
	name_tokens = _name_tokens(source_record.get("business_name"))
	name_token = _significant_name_token(source_record.get("business_name"))
	address_tokens = _address_tokens(source_record.get("business_address"))
	address_numbers = _address_numbers(source_record.get("business_address"))
	keys = (
		("name_tokens", (country, name_tokens)),
		("name_prefix", (country, name_token[:prefix_length]) if name_token else ()),
		("address_tokens", (country, address_tokens)),
		("address_numeric", (country, address_numbers) if address_numbers else ()),
	)
	candidates = set()
	for index_name, key in keys:
		if key:
			candidates.update(indexes[index_name].get(key, ()))
	return candidates


if __name__ == "__main__":
	target_records = [
		{
			"entity_id": "S2-1",
			"business_name": "Acme Coffee Co.",
			"business_address": "12 Main Street",
			"country": "US",
		},
		{
			"entity_id": "S2-2",
			"business_name": "Other Shop",
			"business_address": "12 Main Street",
			"country": "India",
		},
	]
	source_record = {
		"entity_id": "S1-1",
		"business_name": "ACME COFFEE CO",
		"business_address": "12 Main Street",
		"country": "US",
	}

	name_index = build_name_index(target_records)
	address_index = build_address_index(target_records)
	assert generate_candidates(source_record, name_index, address_index) == {"S2-1"}
	assert build_address_index(
		[{"entity_id": "S2-3", "business_address": "", "country": "US"}]
	) == {}
	print(generate_candidates(source_record, name_index, address_index))
