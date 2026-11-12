# PizzaByte - Group Assignment 2

**Presentation** [https://canva.link/q0kznfnzrrz3lg9](https://canva.link/q0kznfnzrrz3lg9)

An Inside Airbnb lakehouse on Databricks Free Edition. Region: **California**;
currency: **USD**. Source: detailed `listings.csv.gz`, Inside Airbnb, CC BY 4.0.

| Role | City | Local archive | Row-level scrape dates recorded in the EDA |
|---|---|---|---|
| A - development/training | Los Angeles | `data/listings_Los_Angeles.csv.gz` | 2026-09-14 to 2026-09-16 |
| B - development/training | San Francisco | `data/listings_San_Francisco.csv.gz` | 2026-09-14 |
| C - unseen transfer evaluation | Santa Clara County | `data/listings_Santa_Clara_County.csv.gz` | Reserved for transfer evaluation; not profiled |

Dates refer to row-level `last_scraped` values in the saved EDA, rather than
independently verified archive publication dates. City C remains unseen until
the A/B pipeline and model are frozen.

## Deliverables currently in this repository

- **3.1:** [Exploratory analysis](notebooks/exploratory_analysis.ipynb).
- **3.2:** [Solution architecture](docs/solution_architecture.md), including the
  [diagram](docs/solution_architecture.svg).
- **3.3:** [ETL notebook](notebooks/etl_pipeline.ipynb),
  [reusable transformations](src/etl_pipeline.py), and
  [serverless Job definition](databricks.yml).
- **3.4:** [Price model](notebooks/price_model.ipynb) and
  [model implementation](src/price_model.py).

## Batch ETL (3.3)

The notebook runs one city per execution. Parameters are `city` (default
`Los_Angeles`), `catalog` (`workspace`), `schema` (`airbnb`), `repo_root`
(auto-detected), `source_dir` (the repo's `data/`), `allow_city_c` (`false`), and
`verify_rerun` (`false`). A workspace Job can run this same notebook with a
different `city`; `databricks.yml` declares the equivalent serverless Job for
bundle deployment, with one concurrent run and no recurring schedule.

The output tables are `workspace.airbnb.bronze_listings`,
`workspace.airbnb.silver_listings` and `workspace.airbnb.gold_model_features`.
Archives are stored unchanged under `/Volumes/workspace/airbnb/raw/`, with
city/content-hash subdirectories. Each refresh replaces only its city through
Delta `replaceWhere`; other cities are preserved. `verify_rerun=true` checks
stable A/B content signatures before and after a repeated run.

Silver uses the EDA's types and derived fields and preserves the raw model
inputs in `price_model.REQUIRED_COLUMNS`. Invalid labels and placeholders stay
in the dataset with `price_valid=false`; `price_class` is a fixed eligibility
classification (`valid`, `no_price`, `placeholder`), not a fitted statistical
outlier fence. Imputation, category grouping/encoding and model fitting are
outside the ETL. Duplicate IDs are resolved by scrape date and source-record
hash; missing/unparseable IDs fail publication.

The model's current configuration defaults to the separate EDA namespace.
`SOURCE_TABLE=workspace.airbnb.silver_listings` selects the pipeline output;
the architecture's shared prediction interface is
`workspace.airbnb.gold_predictions`. Prediction/metric publication remains with
the model, and the five analytical gold tables remain with dashboard/Genie.
The model's own evaluation table and the analytical gold metrics table have
separate writers; changing the prediction destination does not imply moving
the model's evaluation table into the analytical table's name.

The C gate is checked before any C file access. The archive is excluded from
default bundle synchronisation; after the A/B freeze, `source_dir` can refer to
its approved archive location without changing the pipeline code.

Local transformation tests use synthetic data and PySpark:

```bash
python -m unittest discover -s tests -v
```

Databricks execution evidence consists of the selected-city summary, schema
and price-class reports, published A/B counts and optional rerun signatures in
the ETL notebook, together with successful city-parameterised Job runs.

## Platform and scope

Execution uses Databricks Free Edition serverless compute, Unity Catalog
volumes for raw archives, and managed Delta tables for the medallion layers.
The architecture specifies the planned pipeline and model interfaces;
implementation and execution evidence belong to the subsequent deliverables.
