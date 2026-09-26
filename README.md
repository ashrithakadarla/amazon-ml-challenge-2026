# Amazon ML Challenge 2026

Team project for the Amazon ML Challenge 2026.

## Problem

Entity resolution / record linkage across three business data sources.

The goal is to identify Source 2 and Source 3 records corresponding to each Source 1 entity.

## Pipeline

Data
→ Preprocessing
→ Candidate Generation
→ Similarity Features
→ ML Matching
→ Evaluation
→ Final Predictions

## Team Responsibilities

| Member | Responsibility |
|---|---|
| Member 1 | Candidate Generation, Architecture, Integration |
| Member 2 | Data Analysis, Ground Truth |
| Member 3 | Similarity Features, Feature Engineering |
| Member 4 | ML Model, F0.5 Evaluation |

## Repository Structure

```text
src/
├── preprocessing/
├── candidate_generation/
├── features/
├── model/
└── evaluation/

experiments/
docs/
submissions/