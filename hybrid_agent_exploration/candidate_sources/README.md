# PSC passivator seed candidate sources

This directory contains small, tracked CSV seed libraries for PSC additive or
passivator candidate screening. Source files must satisfy
`candidate-library-v1` after normalization with
`screening.candidate_library_builder.CandidateLibraryBuilder`.

## Source policy

- Use CSV only for tracked seed sources.
- Include only real molecules with externally verifiable identity evidence.
- Set `verification_status=verified` for every row.
- Include structured `verification_sources` JSON with at least one
  `kind=molecule` PubChem source containing `pubchem_id` and `url`.
- Add PSC relevance evidence as PubMed, DOI, publisher, or other durable
  literature URLs when available.
- Do not include placeholder URLs such as `example.com`.
- Do not invent vendor catalog IDs or availability claims. Use conservative
  readiness labels such as `pubchem_listed`, `literature_reported`, or
  `not_assessed` unless a source explicitly supports a stronger status.

## Research package usage

Run the package with the tracked seed source by passing `--candidate-source`:

```bash
PYTHONPATH=src python src/run_research_package.py \
  --input path/to/verified_dataset.csv \
  --output results/research_package_with_psc_seed_candidates \
  --dataset-id psc-package-with-seed-candidates \
  --candidate-source candidate_sources/psc_passivator_seed_candidates.csv \
  --candidate-source-name psc-passivator-seed
```

`run_research_package` normalizes the CSV into `candidate-library-v1` artifacts
before the verified candidate discovery stage. The builder is offline-only and
does not re-verify source claims during the run.
