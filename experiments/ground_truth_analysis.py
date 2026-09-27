import pandas as pd
from collections import Counter

GROUND_TRUTH_PATH = (
    "data/student_resource/dataset/train/train_ground_truth.tsv"
)

CHUNK_SIZE = 200_000


def analyze_ground_truth():
    total_rows = 0
    zero_match_count = 0

    match_count_distribution = Counter()
    s2_match_total = 0
    s3_match_total = 0

    only_s2_count = 0
    only_s3_count = 0
    both_s2_s3_count = 0

    max_matches = 0
    max_match_entities = []

    for chunk in pd.read_csv(
        GROUND_TRUTH_PATH,
        sep="\t",
        dtype=str,
        chunksize=CHUNK_SIZE,
    ):
        for _, row in chunk.iterrows():
            source1_id = row["source1_entity_id"]
            matched_ids = str(row["matched_entity_ids"]).strip()

            total_rows += 1

            if not matched_ids or matched_ids.lower() == "nan":
                zero_match_count += 1
                match_count_distribution[0] += 1
                continue

            matches = [
                match.strip()
                for match in matched_ids.split(",")
                if match.strip()
            ]

            match_count = len(matches)
            match_count_distribution[match_count] += 1

            s2_matches = sum(
                1 for match in matches if match.startswith("S2-")
            )
            s3_matches = sum(
                1 for match in matches if match.startswith("S3-")
            )

            s2_match_total += s2_matches
            s3_match_total += s3_matches

            if s2_matches > 0 and s3_matches == 0:
                only_s2_count += 1
            elif s3_matches > 0 and s2_matches == 0:
                only_s3_count += 1
            elif s2_matches > 0 and s3_matches > 0:
                both_s2_s3_count += 1

            if match_count > max_matches:
                max_matches = match_count
                max_match_entities = [source1_id]
            elif match_count == max_matches:
                max_match_entities.append(source1_id)

    print("\n" + "=" * 60)
    print("GROUND TRUTH ANALYSIS")
    print("=" * 60)

    print(f"\nTotal Source 1 entities: {total_rows:,}")
    print(f"Entities with zero matches: {zero_match_count:,}")

    print("\n--- Match Count Distribution ---")

    for count in sorted(match_count_distribution):
        entities = match_count_distribution[count]
        percentage = (entities / total_rows) * 100

        print(
            f"{count:>3} matches : "
            f"{entities:>10,} entities "
            f"({percentage:.2f}%)"
        )

    print("\n--- Total Match Links ---")
    print(f"S2 matches: {s2_match_total:,}")
    print(f"S3 matches: {s3_match_total:,}")
    print(
        f"Total matches: "
        f"{s2_match_total + s3_match_total:,}"
    )

    print("\n--- Match Type Distribution ---")
    print(f"Only S2       : {only_s2_count:,}")
    print(f"Only S3       : {only_s3_count:,}")
    print(f"Both S2 + S3  : {both_s2_s3_count:,}")

    print("\n--- Maximum Matches ---")
    print(f"Maximum matches for one S1: {max_matches}")
    print(
        f"Number of S1 entities with {max_matches} matches: "
        f"{len(max_match_entities):,}"
    )

    if max_match_entities:
        print("\nExample S1 entities with maximum matches:")
        for entity_id in max_match_entities[:10]:
            print(f"  {entity_id}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    analyze_ground_truth()