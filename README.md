# PizzaByte - Group Assignment 2

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

## Platform and scope

Execution uses Databricks Free Edition serverless compute, Unity Catalog
volumes for raw archives, and managed Delta tables for the medallion layers.
The architecture specifies the planned pipeline and model interfaces;
implementation and execution evidence belong to the subsequent deliverables.
