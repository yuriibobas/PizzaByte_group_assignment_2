# Genie space setup (3.6)

The Genie space sits on the gold tables built by `notebooks/gold_dashboard_genie.ipynb`. Run that notebook first. Genie can't be set up from a file in the repo, so this page holds everything we paste into the UI, and we keep it here so the setup can be reproduced.

## 1. Create the space

Genie > New, pick a serverless SQL warehouse, and add these tables from `workspace.airbnb`:

- `gold_listings`: one row per listing, all cities
- `gold_city_summary`: one row per city
- `gold_neighbourhood_stats`: one row per city and neighbourhood
- `gold_model_errors`: one row per listing and model run, with the error
- `gold_model_metrics`: MAE of the model and the baseline per city

Leave silver and the raw tables out. Fewer tables means Genie picks the right one more often, and the gold tables already have the valid-price rule built in.

Title: `Airbnb California: prices and price model`

## 2. General instructions

Paste this into Instructions > Text:

```
Data: Inside Airbnb listings for three California cities, one snapshot per city. One row in gold_listings is one listing. All prices are USD per night.

Cities: "City A" is Los Angeles, "City B" is San Francisco, "City C" is Santa Clara County (the new client, not used to train the model). Filter on city_role ('A', 'B', 'C') when the user says City A/B/C, and on city_name when they name the city. Show city_name in answers, not city.

Prices:
- price is already NULL for listings with no price or a fake placeholder price. Never filter on price_class to "clean" prices; just ignore NULLs.
- Use the median, not the average, for prices. Prices are very skewed.
- Short stays and monthly rentals (stay_type = 'monthly', minimum 28 nights) have different price levels. If the user does not say which, answer for short stays and also give the monthly figure in a separate row. Do not mix them in one number.
- "Entire home" means room_type = 'Entire home/apt'.

Neighbourhoods: names only exist within one city. Always group by city_name and neighbourhood together. When ranking neighbourhoods by price, only use neighbourhoods with at least 30 priced listings, and say so.

Model:
- Use gold_model_errors and gold_model_metrics. Filter on is_evaluation = true unless the user asks about training data. These are the listings the model never trained on: the test split of City A and B, and all priced listings of City C (split = 'new_data').
- If the user does not name a model run, use is_latest_run = true in gold_model_errors (or the highest model_run in gold_model_metrics). price_model_v1 is the model at the end of week 1; a later version (price_model_v2) is the model after the city C changes.
- error = predicted_price - actual_price. Under-priced means error < 0 (the model suggests less than the listing asks). Over-priced means error > 0.
- MAE = average of abs_error. The baseline is the median training price for the same room_type and accommodates.
- For "where" questions about the model, group by city_name and neighbourhood (or room_type) and only keep groups with at least 20 test listings.

Answer style: round prices to whole dollars, state the city and the number of listings behind each figure.
```

## 3. Sample questions

Add these under Settings > Sample questions (they show up as buttons for the user):

- What is the median price of entire homes in City A?
- Which neighbourhoods in City B have the most listings?
- Where does the model under-price listings the most?
- How much better is the model than the baseline in each city?
- Compare the median short-stay price for 2-guest entire homes in City A and City B.
- What share of listings in each city are monthly rentals?

## 4. Example SQL queries

Add these under Instructions > SQL queries, with the question as the title. Genie copies the pattern for similar questions, which matters most for the median and the under-pricing sign.

**What is the median price of entire homes in City A?**

```sql
SELECT city_name, stay_type,
       round(median(price)) AS median_price_usd,
       count(price)         AS listings_with_price
FROM workspace.airbnb.gold_listings
WHERE city_role = 'A'
  AND room_type = 'Entire home/apt'
  AND price IS NOT NULL
GROUP BY city_name, stay_type
ORDER BY stay_type DESC
```

**Which neighbourhoods in City B have the most listings?**

```sql
SELECT city_name, neighbourhood, listings, median_price_short_stay
FROM workspace.airbnb.gold_neighbourhood_stats
WHERE city_role = 'B'
ORDER BY listings DESC
LIMIT 10
```

**Where does the model under-price listings the most?**

```sql
SELECT city_name, neighbourhood,
       count(*)                  AS test_listings,
       round(avg(error))         AS mean_error_usd,
       round(avg(abs_error))     AS mae_usd,
       round(median(error_pct))  AS median_error_pct
FROM workspace.airbnb.gold_model_errors
WHERE is_evaluation AND is_latest_run
GROUP BY city_name, neighbourhood
HAVING count(*) >= 20
ORDER BY mean_error_usd ASC
LIMIT 10
```

**How much better is the model than the baseline in each city?**

```sql
SELECT model_run, scope, n_listings, baseline_mae, model_mae, pct_better_than_baseline
FROM workspace.airbnb.gold_model_metrics
WHERE is_evaluation
ORDER BY model_run, city_role
```

## 5. Check it before the demo

Ask the three questions from the assignment in a new chat and compare the answer with the SQL above run in the SQL editor. Write the result here so we know it was checked:

| Question | Genie answer | SQL answer | Same? |
|---|---|---|---|
| Median price of entire homes in City A | | | |
| Neighbourhoods in City B with the most listings | | | |
| Where the model under-prices most | | | |

If Genie gets one wrong, look at the SQL it generated (Show code) first. Usually the fix is one more sentence in the instructions or one more example query, not a new table.

## What we did to make Genie answer correctly (for the presentation)

- **Gold tables only, five of them.** No silver, no raw. Genie has less to choose from.
- **Bad prices are already NULL.** Placeholder and missing prices can't leak into a median, whatever SQL Genie writes.
- **An `is_evaluation` flag.** A and B are evaluated on their test split, city C on all its listings (`new_data`). One flag covers both, so Genie doesn't have to know the split names.
- **A `city_role` column.** "City A" is not in the data, so we added A/B/C as a column instead of hoping Genie maps the names.
- **Column comments in Unity Catalog** (set in the notebook). Genie reads them, so the sign of `error` and the meaning of `stay_type` are explained where the column is.
- **Stay type in the instructions.** Without it Genie mixes monthly rentals into the median, and since they are priced lower per night the answer comes out too low.
- **Example SQL for the questions where the logic is easy to get backwards** (under-pricing = negative error, median instead of average).
