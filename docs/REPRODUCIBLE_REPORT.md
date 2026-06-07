# Reproducible Report Guide

The main public artifact should be a reproducible EDA report for the Perovskite Database.

## Required provenance

- Data source URL or DOI.
- Data version or download date.
- Input file path.
- Filtering rules.
- Target metric.
- Random seed.
- Train/validation/test split policy when models are included.

## Required outputs

- Main EDA report.
- Supporting information or method notes.
- Key figures with source script or notebook.
- Metric table for any model claim.
- Known limitations.

## Acceptance check

From a clean checkout with the data available, `make research-package` should regenerate the reviewed report artifacts or fail with a clear missing-data message.
