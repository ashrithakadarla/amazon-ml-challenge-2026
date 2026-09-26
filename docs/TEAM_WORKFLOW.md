# Team Workflow

## Overall Pipeline

Data Understanding
        ↓
Preprocessing
        ↓
Candidate Generation
        ↓
Similarity Features
        ↓
ML Matching
        ↓
Evaluation
        ↓
Final Integration
        ↓
Submission

---

## Member 1 — Candidate Generation + Architecture + Integration

### Responsibilities

- Overall pipeline architecture
- Data normalization/preprocessing
- Candidate generation/blocking
- S1 → S2 candidate generation
- S1 → S3 candidate generation
- Candidate recall evaluation
- Integration of all components
- Final inference pipeline
- Final submission generation and validation

### Main Output

Candidate pairs that contain the likely matching records for each Source 1 entity.

---

## Member 2 — Data Analysis + Ground Truth

### Responsibilities

- Dataset profiling
- Missing-value analysis
- Duplicate analysis
- Name/address/country patterns
- Ground-truth analysis
- Number of matches per Source 1
- S2 vs S3 match distribution
- Only-S2 / Only-S3 / Both analysis

### Main Output

Data findings and observations that help design the matching pipeline.

---

## Member 3 — Similarity + Feature Engineering

### Responsibilities

- Business-name similarity
- Address similarity
- Token overlap
- Fuzzy similarity
- Country matching
- Other useful field-level similarity features

### Main Output

Feature-generation pipeline for candidate pairs.

---

## Member 4 — ML Model + F0.5 Evaluation

### Responsibilities

- Build baseline matching model
- Train using candidate pairs and ground truth
- Precision evaluation
- Recall evaluation
- F0.5 evaluation
- Threshold experiments
- Model/inference pipeline

### Main Output

Matching model and selected validation threshold.

---

# Handoff Flow

Member 2
    ↓
Data findings
    ↓
Member 1
    ↓
Candidate pairs
    ↓
Member 3
    ↓
Similarity features
    ↓
Member 4
    ↓
Model + threshold
    ↓
Member 1
    ↓
Final integrated pipeline
    ↓
Test predictions
    ↓
Submission files