# Dashboard (3.5)

`airbnb_dashboard.lvdash.json` is the AI/BI dashboard. It reads only the gold tables built by `notebooks/gold_dashboard_genie.ipynb`, so run that notebook first.

**Import:** Dashboards > Create dashboard > Import dashboard from file, pick the `.lvdash.json`. Or open the file from the repo folder in the workspace; Databricks shows `.lvdash.json` files as dashboards. Then pick a SQL warehouse and publish.

The tables are hard-coded as `workspace.airbnb.gold_*`. If our schema ends up with another name, search and replace it in the JSON before importing.

**Pages**

1. **City overview**: one city at a time (filter at the top, Los Angeles by default). Listings, hosts, median price split into short stays and monthly rentals, availability, ratings, room-type mix and the 15 most expensive neighbourhoods.
2. **Cross-city comparison**: all cities side by side. The main point is price at equal size, because the pooled medians hide that SF listings are smaller.
3. **Model**: baseline against model MAE, predicted against actual, error by room type and neighbourhood, and the 15 most under- and over-priced listings per city. All of it uses rows the model never trained on (`is_evaluation`): the test split of A and B, and every priced listing of city C. Filters for city and model run (`price_model_v1` = end of week 1, a later version after the city C changes).

City C needs no change to the dashboard: every query groups by city, so it shows up after the pipeline and model have run on it.

**Still to do after the model run**

- The Model page title states the A/B result from the model notebook ($115 against $189 baseline MAE). After the city C run, add what happened on C to it.
- The titles of the first two pages come from the EDA on A and B. Check them against the published dashboard.
- The JSON was written by hand and checked against the gold tables with a local Spark run, but not yet imported into Databricks. If a widget looks off after import, fix it in the editor and export the JSON again over this file.
