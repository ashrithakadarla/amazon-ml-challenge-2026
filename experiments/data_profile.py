from collections import Counter
from pathlib import Path

import pandas as pd


CHUNK_SIZE = 200_000
SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
DATASET_DIR = (
	Path(__file__).resolve().parents[1] / "data" / "student_resource" / "dataset" / "train"
)
SOURCE_FILES = [
	DATASET_DIR / "train_source1.tsv",
	DATASET_DIR / "train_source2.tsv",
	DATASET_DIR / "train_source3.tsv",
]


def update_length_counts(length_counts, values):
	"""Add string lengths from one chunk to an aggregate frequency table."""
	length_counts.update(values.str.len().value_counts().to_dict())


def median_from_length_counts(length_counts, total_count):
	"""Return the exact median using the frequency table of observed lengths."""
	if total_count == 0:
		return 0.0

	middle_positions = ((total_count + 1) // 2, (total_count + 2) // 2)
	cumulative_count = 0
	middle_values = []
	for length in sorted(length_counts):
		cumulative_count += length_counts[length]
		while middle_positions and cumulative_count >= middle_positions[0]:
			middle_values.append(length)
			middle_positions = middle_positions[1:]
	return sum(middle_values) / len(middle_values)


def length_statistics(length_counts, total_count):
	"""Calculate min, max, mean, and median from aggregated length counts."""
	if total_count == 0:
		return {"minimum": 0, "maximum": 0, "mean": 0.0, "median": 0.0}

	total_length = sum(length * count for length, count in length_counts.items())
	return {
		"minimum": min(length_counts),
		"maximum": max(length_counts),
		"mean": total_length / total_count,
		"median": median_from_length_counts(length_counts, total_count),
	}


def profile_source(source_path):
	"""Profile one source file while retaining only aggregate state."""
	total_rows = 0
	empty_name_count = 0
	empty_address_count = 0
	country_counts = Counter()
	name_length_counts = Counter()
	address_length_counts = Counter()
	samples = []

	reader = pd.read_csv(
		source_path,
		sep="\t",
		dtype=str,
		keep_default_na=False,
		chunksize=CHUNK_SIZE,
	)
	for chunk in reader:
		total_rows += len(chunk)
		names = chunk["business_name"]
		addresses = chunk["business_address"]

		empty_name_count += names.eq("").sum()
		empty_address_count += addresses.eq("").sum()
		country_counts.update(chunk["country"])
		update_length_counts(name_length_counts, names)
		update_length_counts(address_length_counts, addresses)

		if len(samples) < 5:
			non_empty = chunk.loc[
				names.ne("") & addresses.ne(""), SOURCE_COLUMNS
			]
			samples.extend(non_empty.head(5 - len(samples)).to_dict("records"))

	return {
		"total_rows": total_rows,
		"empty_name_count": empty_name_count,
		"empty_address_count": empty_address_count,
		"country_counts": country_counts,
		"name_lengths": length_statistics(name_length_counts, total_rows),
		"address_lengths": length_statistics(address_length_counts, total_rows),
		"samples": samples,
	}


def percentage(count, total):
	return (count / total * 100) if total else 0.0


def print_length_statistics(label, statistics):
	print(f"{label} length statistics:")
	print(f"  Minimum: {statistics['minimum']}")
	print(f"  Maximum: {statistics['maximum']}")
	print(f"  Mean: {statistics['mean']:.2f}")
	print(f"  Median: {statistics['median']:.2f}")


def print_profile(source_path, profile):
	total_rows = profile["total_rows"]
	print(f"\n{'=' * 80}\n{source_path.name}\n{'=' * 80}")
	print(f"Total rows: {total_rows:,}")
	print(
		"Empty business_name: "
		f"{profile['empty_name_count']:,} "
		f"({percentage(profile['empty_name_count'], total_rows):.2f}%)"
	)
	print(
		"Empty business_address: "
		f"{profile['empty_address_count']:,} "
		f"({percentage(profile['empty_address_count'], total_rows):.2f}%)"
	)

	print("Country frequency distribution (top 20):")
	for country, count in profile["country_counts"].most_common(20):
		print(f"  {country or '<empty>'}: {count:,} ({percentage(count, total_rows):.2f}%)")

	print_length_statistics("Business name", profile["name_lengths"])
	print_length_statistics("Business address", profile["address_lengths"])

	print("Five sample non-empty records:")
	for sample in profile["samples"]:
		print(f"  {sample}")


def main():
	for source_path in SOURCE_FILES:
		print(f"Processing {source_path} in chunks of {CHUNK_SIZE:,} rows...")
		print_profile(source_path, profile_source(source_path))


if __name__ == "__main__":
	main()
