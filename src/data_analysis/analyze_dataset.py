from pathlib import Path
import pandas as pd


# Original challenge dataset location.
# We only READ from this directory.
DATA_DIR = Path(r"D:\Dataset\student_resource\dataset\train")

# Number of rows processed at a time.
CHUNK_SIZE = 100_000


def analyze_missing_values(file_name):
    """Calculate missing and empty values using chunked reading."""

    file_path = DATA_DIR / file_name

    total_rows = 0
    missing_counts = {
        "business_name": 0,
        "business_address": 0,
        "country": 0,
    }

    empty_counts = {
        "business_name": 0,
        "business_address": 0,
        "country": 0,
    }

    for chunk in pd.read_csv(
        file_path,
        sep="\t",
        chunksize=CHUNK_SIZE,
        keep_default_na=True,
    ):

        total_rows += len(chunk)

        for column in missing_counts:

            # Actual NaN values
            missing_counts[column] += chunk[column].isna().sum()

            # Empty strings / whitespace-only strings
            values = chunk[column].fillna("").astype(str).str.strip()

            empty_counts[column] += values.eq("").sum()

    print("\n" + "=" * 70)
    print(f"FILE: {file_name}")
    print("=" * 70)

    print(f"Total rows: {total_rows:,}")

    print("\nMissing values:")
    for column, count in missing_counts.items():
        percentage = (count / total_rows) * 100
        print(
            f"{column}: "
            f"{count:,} "
            f"({percentage:.4f}%)"
        )

    print("\nEmpty / whitespace-only values:")
    for column, count in empty_counts.items():
        percentage = (count / total_rows) * 100
        print(
            f"{column}: "
            f"{count:,} "
            f"({percentage:.4f}%)"
        )

def analyze_duplicates(file_name):
    """Analyze exact duplicate records and duplicate field values."""

    file_path = DATA_DIR / file_name

    total_rows = 0

    # Track exact complete records.
    exact_records = set()

    # Track names and addresses.
    name_counts = {}
    address_counts = {}

    for chunk in pd.read_csv(
        file_path,
        sep="\t",
        chunksize=CHUNK_SIZE,
        keep_default_na=True,
    ):

        total_rows += len(chunk)

        # ---------------------------------------------------------
        # Exact duplicate records
        # ---------------------------------------------------------

        records = chunk[
            [
                "business_name",
                "business_address",
                "country",
            ]
        ].fillna("")

        for row in records.itertuples(index=False, name=None):
            exact_records.add(row)

        # ---------------------------------------------------------
        # Business-name frequencies
        # ---------------------------------------------------------

        names = (
            chunk["business_name"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        for name, count in names.value_counts().items():

            if name:
                name_counts[name] = (
                    name_counts.get(name, 0) + int(count)
                )

        # ---------------------------------------------------------
        # Address frequencies
        # ---------------------------------------------------------

        addresses = (
            chunk["business_address"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        for address, count in addresses.value_counts().items():

            if address:
                address_counts[address] = (
                    address_counts.get(address, 0) + int(count)
                )

    # -------------------------------------------------------------
    # Final statistics
    # -------------------------------------------------------------

    exact_unique_records = len(exact_records)

    exact_duplicate_rows = (
        total_rows - exact_unique_records
    )

    duplicate_name_values = sum(
        1 for count in name_counts.values()
        if count > 1
    )

    duplicate_name_rows = sum(
        count for count in name_counts.values()
        if count > 1
    )

    duplicate_address_values = sum(
        1 for count in address_counts.values()
        if count > 1
    )

    duplicate_address_rows = sum(
        count for count in address_counts.values()
        if count > 1
    )

    print("\n" + "=" * 70)
    print(f"DUPLICATE ANALYSIS: {file_name}")
    print("=" * 70)

    print(f"Total rows: {total_rows:,}")

    print("\nExact complete-record duplicates:")
    print(f"Unique records: {exact_unique_records:,}")
    print(f"Duplicate rows: {exact_duplicate_rows:,}")

    print("\nDuplicate business names:")
    print(f"Repeated name values: {duplicate_name_values:,}")
    print(f"Rows belonging to repeated names: {duplicate_name_rows:,}")

    print("\nDuplicate addresses:")
    print(f"Repeated address values: {duplicate_address_values:,}")
    print(f"Rows belonging to repeated addresses: {duplicate_address_rows:,}")

def analyze_ground_truth():
    """Analyze the distribution of matches in the ground truth."""

    file_path = DATA_DIR / "train_ground_truth.tsv"

    total_s1 = 0
    match_counts = {}

    s2_only = 0
    s3_only = 0
    both_s2_s3 = 0
    no_matches = 0

    for chunk in pd.read_csv(
        file_path,
        sep="\t",
        chunksize=CHUNK_SIZE,
        keep_default_na=True,
    ):

        for matched_ids in chunk["matched_entity_ids"]:

            total_s1 += 1

            # Handle empty / missing match lists
            if pd.isna(matched_ids) or str(matched_ids).strip() == "":
                match_count = 0
                no_matches += 1
            else:
                matched_ids = str(matched_ids).strip()

                ids = [
                    x.strip()
                    for x in matched_ids.split(",")
                    if x.strip()
                ]

                match_count = len(ids)

                has_s2 = any(x.startswith("S2-") for x in ids)
                has_s3 = any(x.startswith("S3-") for x in ids)

                if has_s2 and has_s3:
                    both_s2_s3 += 1
                elif has_s2:
                    s2_only += 1
                elif has_s3:
                    s3_only += 1

            match_counts[match_count] = (
                match_counts.get(match_count, 0) + 1
            )

    maximum_matches = max(match_counts.keys())

    print("\n" + "=" * 70)
    print("GROUND TRUTH ANALYSIS")
    print("=" * 70)

    print(f"Total Source 1 entities: {total_s1:,}")

    print("\nMatch-count distribution:")

    for count in sorted(match_counts):
        rows = match_counts[count]
        percentage = (rows / total_s1) * 100

        print(
            f"{count} matches: "
            f"{rows:,} "
            f"({percentage:.4f}%)"
        )

    print("\nMaximum matches for one Source 1 entity:")
    print(maximum_matches)

    print("\nSource distribution:")

    print(f"No matches: {no_matches:,}")
    print(f"S2 only: {s2_only:,}")
    print(f"S3 only: {s3_only:,}")
    print(f"Both S2 and S3: {both_s2_s3:,}")

def analyze_source_match_distribution():
    """Analyze how ground-truth matches are distributed between S2 and S3."""

    file_path = DATA_DIR / "train_ground_truth.tsv"

    total_s1 = 0

    s2_match_counts = {}
    s3_match_counts = {}
    combination_counts = {}

    total_s2_matches = 0
    total_s3_matches = 0

    for chunk in pd.read_csv(
        file_path,
        sep="\t",
        chunksize=CHUNK_SIZE,
        keep_default_na=True,
    ):

        for matched_ids in chunk["matched_entity_ids"]:

            total_s1 += 1

            if pd.isna(matched_ids) or str(matched_ids).strip() == "":
                s2_count = 0
                s3_count = 0

            else:
                ids = [
                    x.strip()
                    for x in str(matched_ids).split(",")
                    if x.strip()
                ]

                s2_count = sum(
                    1 for x in ids if x.startswith("S2-")
                )

                s3_count = sum(
                    1 for x in ids if x.startswith("S3-")
                )

            total_s2_matches += s2_count
            total_s3_matches += s3_count

            s2_match_counts[s2_count] = (
                s2_match_counts.get(s2_count, 0) + 1
            )

            s3_match_counts[s3_count] = (
                s3_match_counts.get(s3_count, 0) + 1
            )

            combination = (s2_count, s3_count)

            combination_counts[combination] = (
                combination_counts.get(combination, 0) + 1
            )

    print("\n" + "=" * 70)
    print("S2 / S3 MATCH DISTRIBUTION")
    print("=" * 70)

    print(f"Total Source 1 entities: {total_s1:,}")

    print("\nTotal ground-truth matches:")
    print(f"S2 matches: {total_s2_matches:,}")
    print(f"S3 matches: {total_s3_matches:,}")
    print(
        f"Total matches: "
        f"{total_s2_matches + total_s3_matches:,}"
    )

    print("\nAverage matches per Source 1 entity:")
    print(
        f"S2: {total_s2_matches / total_s1:.4f}"
    )
    print(
        f"S3: {total_s3_matches / total_s1:.4f}"
    )

    print("\nS2 match-count distribution:")

    for count in sorted(s2_match_counts):
        rows = s2_match_counts[count]
        percentage = (rows / total_s1) * 100

        print(
            f"{count} S2 matches: "
            f"{rows:,} "
            f"({percentage:.4f}%)"
        )

    print("\nS3 match-count distribution:")

    for count in sorted(s3_match_counts):
        rows = s3_match_counts[count]
        percentage = (rows / total_s1) * 100

        print(
            f"{count} S3 matches: "
            f"{rows:,} "
            f"({percentage:.4f}%)"
        )

    print("\nS2 / S3 combination distribution:")

    for (s2_count, s3_count), rows in sorted(
        combination_counts.items()
    ):
        percentage = (rows / total_s1) * 100

        print(
            f"S2={s2_count}, "
            f"S3={s3_count}: "
            f"{rows:,} "
            f"({percentage:.4f}%)"
        )

def analyze_text_lengths(file_name):
    """Analyze business name and address length statistics."""

    file_path = DATA_DIR / file_name

    total_rows = 0

    name_lengths = []
    address_lengths = []

    # Store only summary statistics instead of all strings.
    name_min = float("inf")
    name_max = 0
    address_min = float("inf")
    address_max = 0

    name_total_length = 0
    address_total_length = 0

    short_names = 0
    long_names = 0
    short_addresses = 0
    long_addresses = 0

    for chunk in pd.read_csv(
        file_path,
        sep="\t",
        chunksize=CHUNK_SIZE,
        keep_default_na=True,
    ):

        total_rows += len(chunk)

        names = (
            chunk["business_name"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        addresses = (
            chunk["business_address"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        name_len = names.str.len()
        address_len = addresses.str.len()

        # Overall statistics
        name_total_length += name_len.sum()
        address_total_length += address_len.sum()

        name_min = min(name_min, name_len.min())
        name_max = max(name_max, name_len.max())

        address_min = min(address_min, address_len.min())
        address_max = max(address_max, address_len.max())

        # Very short names / addresses
        short_names += (name_len <= 3).sum()
        short_addresses += (address_len <= 10).sum()

        # Very long names / addresses
        long_names += (name_len >= 100).sum()
        long_addresses += (address_len >= 200).sum()

    print("\n" + "=" * 70)
    print(f"TEXT LENGTH ANALYSIS: {file_name}")
    print("=" * 70)

    print(f"Total rows: {total_rows:,}")

    print("\nBusiness name length:")
    print(f"Minimum: {int(name_min)} characters")
    print(f"Maximum: {int(name_max)} characters")
    print(
        f"Average: "
        f"{name_total_length / total_rows:.2f} characters"
    )

    print("\nBusiness address length:")
    print(f"Minimum: {int(address_min)} characters")
    print(f"Maximum: {int(address_max)} characters")
    print(
        f"Average: "
        f"{address_total_length / total_rows:.2f} characters"
    )

    print("\nPotentially difficult text cases:")

    print(
        f"Very short names (<= 3 chars): "
        f"{short_names:,}"
    )

    print(
        f"Long names (>= 100 chars): "
        f"{long_names:,}"
    )

    print(
        f"Very short addresses (<= 10 chars): "
        f"{short_addresses:,}"
    )

    print(
        f"Long addresses (>= 200 chars): "
        f"{long_addresses:,}"
    )

def analyze_country_distribution(file_name):
    """Analyze country frequency distribution."""

    file_path = DATA_DIR / file_name

    country_counts = {}
    total_rows = 0

    for chunk in pd.read_csv(
        file_path,
        sep="\t",
        chunksize=CHUNK_SIZE,
        keep_default_na=True,
    ):

        total_rows += len(chunk)

        countries = (
            chunk["country"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        for country, count in countries.value_counts().items():
            country_counts[country] = (
                country_counts.get(country, 0) + int(count)
            )

    print("\n" + "=" * 70)
    print(f"COUNTRY DISTRIBUTION: {file_name}")
    print("=" * 70)

    print(f"Total rows: {total_rows:,}")
    print(f"Unique countries: {len(country_counts):,}")

    print("\nCountries:")

    for country, count in sorted(
        country_counts.items(),
        key=lambda x: x[1],
        reverse=True,
    ):
        percentage = (count / total_rows) * 100

        print(
            f"{country}: "
            f"{count:,} "
            f"({percentage:.4f}%)"
        )

def main():
    source_files = [
        "train_source1.tsv",
        "train_source2.tsv",
        "train_source3.tsv",
    ]

    for file_name in source_files:
        analyze_missing_values(file_name)

    for file_name in source_files:
        analyze_duplicates(file_name)

    analyze_ground_truth()
    analyze_source_match_distribution()

    for file_name in source_files:
        analyze_text_lengths(file_name)
    
    for file_name in source_files:
        analyze_country_distribution(file_name)


if __name__ == "__main__":
    main()