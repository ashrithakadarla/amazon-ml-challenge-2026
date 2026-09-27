# Data Analysis

## 1. Overview

The Amazon ML Challenge 2026 dataset contains three source datasets and one
ground-truth mapping file.

### Training dataset

| File | Rows |
|---|---:|
| `train_source1.tsv` | 2,206,821 |
| `train_source2.tsv` | 5,034,616 |
| `train_source3.tsv` | 5,285,603 |
| `train_ground_truth.tsv` | 2,206,821 |

The three source files contain the following fields:

- `entity_id`
- `business_name`
- `business_address`
- `country`

The ground-truth file contains:

- `source1_entity_id`
- `matched_entity_ids`

The source files are tab-separated and were analyzed using `sep="\t"`.

---

## 2. Missing Value Analysis

### Source 1

| Field | Missing |
|---|---:|
| `business_name` | 0 (0.0000%) |
| `business_address` | 0 (0.0000%) |
| `country` | 0 (0.0000%) |

Source 1 contains no missing values in the matching fields.

### Source 2

| Field | Missing |
|---|---:|
| `business_name` | 2 (0.0000%) |
| `business_address` | 168,967 (3.3561%) |
| `country` | 0 (0.0000%) |

### Source 3

| Field | Missing |
|---|---:|
| `business_name` | 13 (0.0002%) |
| `business_address` | 175,916 (3.3282%) |
| `country` | 0 (0.0000%) |

### Findings

- Source 1 is complete for business name, address and country.
- Source 2 and Source 3 have approximately 3.3% missing addresses.
- Business names are almost always present.
- Country is complete across all three sources.
- Matching logic must handle missing addresses without treating them as
  normal text values.

---

## 3. Duplicate Analysis

### Source 1

| Measure | Result |
|---|---:|
| Exact unique records | 2,206,821 |
| Exact duplicate rows | 0 |
| Repeated business-name values | 177,793 |
| Rows belonging to repeated names | 845,385 |
| Repeated address values | 40,089 |
| Rows belonging to repeated addresses | 116,304 |

Source 1 contains no exact duplicate records.

### Source 2

| Measure | Result |
|---|---:|
| Exact unique records | 5,008,743 |
| Exact duplicate rows | 25,873 |
| Repeated business-name values | 239,778 |
| Rows belonging to repeated names | 872,384 |
| Repeated address values | 421,472 |
| Rows belonging to repeated addresses | 949,860 |

### Source 3

| Measure | Result |
|---|---:|
| Exact unique records | 5,266,743 |
| Exact duplicate rows | 18,860 |
| Repeated business-name values | 258,275 |
| Rows belonging to repeated names | 892,257 |
| Repeated address values | 382,890 |
| Rows belonging to repeated addresses | 859,813 |

### Findings

- Source 1 is deduplicated at the complete-record level.
- Source 2 and Source 3 contain some exact duplicate records.
- Repeated business names and addresses are common.
- Name-only or address-only matching can therefore produce ambiguous
  candidate sets.
- Candidate generation should combine multiple signals rather than relying
  exclusively on one field.

---

## 4. Ground-Truth Match Distribution

There are 2,206,821 Source 1 entities in the ground truth.

| Number of matches | S1 entities | Percentage |
|---:|---:|---:|
| 0 | 123,247 | 5.5848% |
| 1 | 119,157 | 5.3995% |
| 2 | 375,212 | 17.0024% |
| 3 | 530,841 | 24.0546% |
| 4 | 484,115 | 21.9372% |
| 5 | 321,957 | 14.5892% |
| 6 | 164,868 | 7.4708% |
| 7 | 63,968 | 2.8986% |
| 8 | 18,680 | 0.8465% |
| 9 | 4,205 | 0.1905% |
| 10 | 534 | 0.0242% |
| 11 | 37 | 0.0017% |

Maximum number of matches for one Source 1 entity: **11**.

### Findings

- 123,247 Source 1 entities have no matches.
- 119,157 have exactly one match.
- 1,964,417 have at least two matches.
- The problem is therefore not restricted to one-to-one matching.
- The matching system must support zero, one, or multiple matches for an
  individual Source 1 entity.

---

## 5. S2 and S3 Match Distribution

Total ground-truth matches:

| Source | Matches | Average per S1 |
|---|---:|---:|
| S2 | 3,693,619 | 1.6737 |
| S3 | 3,944,746 | 1.7875 |
| Total | 7,638,365 | — |

### S2 match-count distribution

| S2 matches | S1 entities | Percentage |
|---:|---:|---:|
| 0 | 287,745 | 13.0389% |
| 1 | 789,108 | 35.7577% |
| 2 | 652,779 | 29.5801% |
| 3 | 333,957 | 15.1329% |
| 4 | 119,078 | 5.3959% |
| 5 | 24,154 | 1.0945% |

### S3 match-count distribution

| S3 matches | S1 entities | Percentage |
|---:|---:|---:|
| 0 | 266,276 | 12.0660% |
| 1 | 716,417 | 32.4638% |
| 2 | 668,375 | 30.2868% |
| 3 | 372,443 | 16.8769% |
| 4 | 145,116 | 6.5758% |
| 5 | 35,378 | 1.6031% |
| 6 | 2,816 | 0.1276% |

### Source coverage per S1

| Match source | S1 entities |
|---|---:|
| No matches | 123,247 |
| S2 only | 143,029 |
| S3 only | 164,498 |
| Both S2 and S3 | 1,776,047 |

### Findings

- S3 has slightly more ground-truth matches than S2.
- Most Source 1 entities that have matches are represented in both S2 and
  S3.
- The system should independently evaluate candidate pairs from both source
  datasets.
- The system should not assume exactly one matching record per Source 1
  entity.

---

## 6. Business Name and Address Length Analysis

### Source 1

| Statistic | Business name | Business address |
|---|---:|---:|
| Minimum | 3 | 11 |
| Maximum | 105 | 256 |
| Average | 24.03 | 52.07 |

Additional cases:

- Names with <=3 characters: 561
- Names with >=100 characters: 2
- Addresses with <=10 characters: 0
- Addresses with >=200 characters: 58

### Source 2

| Statistic | Business name | Business address |
|---|---:|---:|
| Minimum | 0 | 0 |
| Maximum | 104 | 249 |
| Average | 25.10 | 46.23 |

Additional cases:

- Names with <=3 characters: 2,946
- Names with >=100 characters: 3
- Addresses with <=10 characters: 169,024
- Addresses with >=200 characters: 59

### Source 3

| Statistic | Business name | Business address |
|---|---:|---:|
| Minimum | 0 | 0 |
| Maximum | 123 | 240 |
| Average | 25.20 | 46.71 |

Additional cases:

- Names with <=3 characters: 17,006
- Names with >=100 characters: 3
- Addresses with <=10 characters: 176,445
- Addresses with >=200 characters: 62

### Findings

- Business names generally contain around 24–25 characters on average.
- Source 2 and Source 3 contain missing/very short addresses.
- Similarity features should account for missing address information.
- Very short names may provide weak evidence by themselves.
- Address and name similarity should therefore be considered together.

---

## 7. Country Distribution

### Source 1

| Country | Rows | Percentage |
|---|---:|---:|
| US | 1,323,633 | 59.9792% |
| India | 883,188 | 40.0208% |

### Source 2

| Country | Rows | Percentage |
|---|---:|---:|
| US | 3,016,817 | 59.9215% |
| India | 2,017,799 | 40.0785% |

### Source 3

| Country | Rows | Percentage |
|---|---:|---:|
| US | 3,170,056 | 59.9753% |
| India | 2,115,547 | 40.0247% |

### Findings

The training data contains two countries:

- US
- India

The challenge specifies that the country field is an open set and that
France appears in the test data.

Therefore:

- Country should be used as a matching/blocking signal.
- The implementation should not hard-code only US and India.
- The system must support unseen country values.

---

## 8. Implications for Candidate Generation

The analysis suggests the following considerations for candidate generation:

1. Name-only blocking can produce many candidates because business names are
   frequently repeated.
2. Address-only blocking can also produce many candidates because repeated
   addresses are common.
3. Country can provide a useful blocking signal but must support unseen
   countries.
4. Missing addresses in S2 and S3 mean address-based blocking cannot be the
   only candidate-generation strategy.
5. Candidate generation should use multiple blocking strategies to maintain
   recall.
6. The system must support multiple matches for one Source 1 entity.
7. Since some Source 1 entities have no matches, the system must also support
   an empty final match set.

---

## 9. Implications for Similarity Features

The analysis suggests that similarity features should consider:

- Business-name similarity
- Business-address similarity
- Country agreement
- Combined name and address evidence
- Missing-address indicators

Because repeated names and addresses are common, a single similarity signal
should not be treated as sufficient evidence of a match.

---

## 10. Key Conclusions

The training data shows that the entity-resolution task has several important
characteristics:

1. Source 1 is clean and contains no exact duplicate records.
2. Source 2 and Source 3 contain a small number of exact duplicate records.
3. Missing addresses occur in approximately 3.3% of S2 and S3 records.
4. Business names and addresses are frequently repeated.
5. Most Source 1 entities have multiple ground-truth matches.
6. One Source 1 entity can have up to 11 ground-truth matches.
7. Both S2 and S3 contribute substantially to the final matches.
8. The country field is complete in the training sources.
9. Training contains US and India, while the challenge requires support for
   unseen countries such as France.
10. Candidate generation must prioritize recall while controlling the number
    of ambiguous candidates.
11. Matching features should combine name, address and country information
    while handling missing values.

This analysis will be used to guide candidate-generation and similarity-feature
design in the subsequent stages of the project.