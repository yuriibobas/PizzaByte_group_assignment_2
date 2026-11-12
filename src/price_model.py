"""Price model for the Airbnb lakehouse (group assignment 2, deliverable 3.4).

Everything that has to be reused lives here, so the notebook only calls functions:

* the feature contract (REQUIRED_COLUMNS, FEATURES, LEAK_COLUMNS) and ``add_features``;
* the train/test split (``make_split``);
* the baseline (``MedianBaseline``) and the model (``PriceModel``, a ridge or gradient boosting
  pipeline fitted on log price and converted back to dollars);
* metrics, cross-validation and permutation importance;
* ``score_listings``: predictions for every listing in the layout of the gold table
  (also what the first run on city C calls);
* ``save_model`` / ``load_model``.

The input is a pandas DataFrame with one row per listing and the columns in REQUIRED_COLUMNS
(``load_listings_csv`` builds it from the raw files; a Silver table with the same column names works too). Nothing here depends on Spark or on a city name.
"""
from __future__ import annotations

import json
import os
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

SEED = 42
TARGET = "price"                       # nightly price in the region's currency
PLACEHOLDER_PRICE_PER_GUEST = 2500     # a price per guest at or above this is a number typed to block a listing
PLACEHOLDER_PRICE = 99_999

# --------------------------------------------------------------------------- feature contract
REQUIRED_COLUMNS = [
    "city", "id", "room_type", "property_type", "price", "accommodates", "bedrooms", "beds",
    "bathrooms_num", "bathroom_shared", "n_amenities", "minimum_nights", "host_is_superhost",
    "host_listings_count", "calculated_host_listings_count", "hosts_time_as_host_years",
    "has_license", "latitude", "longitude",
]
OPTIONAL_COLUMNS = ["neighbourhood_cleansed"]          # only used to label the output
_TEXT_COLUMNS = ["city", "room_type", "property_type", "neighbourhood_cleansed"]
_NUMERIC_RAW = [c for c in REQUIRED_COLUMNS if c not in ("city", "id", "room_type", "property_type")]

CAT_FEATURES = ["room_type", "property_type"]
NUM_FEATURES = [
    "accommodates", "bedrooms", "beds", "bathrooms_num", "bathroom_shared", "n_amenities",
    "minimum_nights_capped", "is_monthly_rental",
    "host_is_superhost", "log_host_listings_count", "log_calc_host_listings_count",
    "hosts_time_as_host_years", "has_license",
    "dist_to_center_km", "dx_km", "dy_km",
]
# v2 change (city C, 3.7): the raw "latitude" and "longitude" are no longer features. v1 learned price levels from absolute
# coordinates and could not transfer to a city outside the A/B coordinate ranges. The city-relative features stay.
FEATURES = CAT_FEATURES + NUM_FEATURES

# groups used for permutation importance (a group is shuffled as a whole)
FEATURE_GROUPS = {
    "size and capacity": ["accommodates", "bedrooms", "beds", "bathrooms_num", "bathroom_shared"],
    "room type": ["room_type"],
    "property type": ["property_type"],
    "amenities": ["n_amenities"],
    "stay rules": ["minimum_nights_capped", "is_monthly_rental"],
    "host": ["host_is_superhost", "log_host_listings_count", "log_calc_host_listings_count",
             "hosts_time_as_host_years", "has_license"],
    "location": ["dist_to_center_km", "dx_km", "dy_km"],
}

# columns that must never be used to predict price (leakage review, EDA section 14)
LEAK_COLUMNS = [
    "price", "price_text", "price_per_guest", "price_quote_price_per_night", "price_quote_total_price",
    "price_quote_raw", "price_quote_checkin_date", "price_quote_checkout_date", "quote_nights",
    "quote_lead_days", "quote_is_monthly", "quote_is_available", "quote_json_price_per_night",
    "estimated_revenue_l365d", "estimated_occupancy_l365d", "availability_30", "availability_60",
    "availability_90", "availability_365", "availability_eoy", "has_availability",
    "number_of_reviews", "number_of_reviews_ltm", "number_of_reviews_l30d", "reviews_per_month",
    "review_scores_rating", "review_scores_accuracy", "review_scores_cleanliness",
    "review_scores_checkin", "review_scores_communication", "review_scores_location",
    "review_scores_value", "source", "is_stale", "scrape_id", "last_scraped", "id", "host_id",
]

FEATURE_DOC = [
    ("room_type", "categorical", "Entire home, private room, hotel room or shared room", "Property attribute; overlaps with property_type, so its own permutation importance is small"),
    ("property_type", "categorical", "Type of property (the 25 most common in training, the rest grouped as Other)", "Property attribute; grouping avoids a long tail of rare levels"),
    ("accommodates", "number", "Number of guests", "Property attribute; drives price"),
    ("bedrooms", "number", "Bedrooms (0 = studio)", "Property attribute"),
    ("beds", "number", "Beds", "Property attribute"),
    ("bathrooms_num", "number", "Bathrooms parsed from the text (half-bath = 0.5)", "Property attribute"),
    ("bathroom_shared", "0/1", "Bathroom is shared", "Property attribute"),
    ("n_amenities", "number", "Number of amenities listed", "Property attribute; robust summary of a free-text list"),
    ("minimum_nights_capped", "number", "Host's minimum stay, capped at 365", "Booking rule set by the host; carries the quoted stay length"),
    ("is_monthly_rental", "0/1", "Minimum stay of 28 nights or more", "Monthly listings are priced about 40 % lower per night"),
    ("host_is_superhost", "0/1", "Host has the Superhost badge", "Host attribute known before a price is set"),
    ("log_host_listings_count", "number", "log(1 + listings of the host)", "Host attribute (professional operators price differently)"),
    ("log_calc_host_listings_count", "number", "log(1 + listings of the host in this scrape)", "Host attribute"),
    ("hosts_time_as_host_years", "number", "Years since the host joined", "Host attribute"),
    ("has_license", "0/1", "A licence number is shown", "Listing attribute"),
    ("dist_to_center_km", "number", "Distance to the median point of the city's listings", "Location relative to the city, computed from the city's own listings (no price used); the model has no absolute coordinates"),
    ("dx_km", "number", "East-west offset from that point", "Location relative to the city"),
    ("dy_km", "number", "North-south offset from that point", "Location relative to the city"),
]


def assert_no_leakage(features: Iterable[str] = FEATURES) -> None:
    """Raise if a feature is on the leakage list or if the groups do not match the feature list."""
    bad = sorted(set(features) & set(LEAK_COLUMNS))
    if bad:
        raise AssertionError(f"leaking columns used as features: {bad}")
    grouped = sorted(c for cols in FEATURE_GROUPS.values() for c in cols)
    if grouped != sorted(FEATURES):
        raise AssertionError("FEATURE_GROUPS and FEATURES do not contain the same columns")


# --------------------------------------------------------------------------- loading from the raw files
_RAW_COLUMNS = [
    "id", "room_type", "property_type", "neighbourhood_cleansed", "price", "accommodates", "bedrooms", "beds",
    "bathrooms_text", "amenities", "minimum_nights", "host_is_superhost", "host_listings_count",
    "calculated_host_listings_count", "hosts_time_as_host_years", "license", "latitude", "longitude",
]


def _type_raw(r: pd.DataFrame) -> pd.DataFrame:
    """Raw Inside Airbnb columns (as read from the CSV) to the typed columns in REQUIRED_COLUMNS."""
    def col(name: str) -> pd.Series:
        return r[name] if name in r.columns else pd.Series(np.nan, index=r.index)

    def n_amenities(x):
        try:
            return float(len(json.loads(x)))
        except (TypeError, ValueError):
            return np.nan

    out = pd.DataFrame(index=r.index)
    out["city"] = r["city"]
    out["id"] = r["id"].astype("int64")
    for c in ("room_type", "property_type", "neighbourhood_cleansed"):
        out[c] = col(c)
    out["price"] = pd.to_numeric(col("price").astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce").astype("float64")

    has_text = col("bathrooms_text").notna()
    text = col("bathrooms_text").astype(str).str.lower()
    digits = pd.to_numeric(text.str.extract(r"(\d+(?:\.\d+)?)")[0], errors="coerce")
    digits = digits.where(~(text.str.contains("half") & digits.isna()), 0.5)          # "Half-bath" = 0.5
    out["bathrooms_num"] = digits.where(has_text)
    out["bathroom_shared"] = text.str.contains("shared").astype(float).where(has_text)
    out["n_amenities"] = col("amenities").map(n_amenities)
    out["host_is_superhost"] = col("host_is_superhost").map({"t": 1.0, "f": 0.0})
    out["has_license"] = col("license").notna().astype(float)
    for c in ("accommodates", "bedrooms", "beds", "minimum_nights", "host_listings_count",
              "calculated_host_listings_count", "hosts_time_as_host_years", "latitude", "longitude"):
        out[c] = pd.to_numeric(col(c), errors="coerce").astype("float64")
    return out


def load_listings_csv(data_dir: str, cities: Sequence[str], file_pattern: str = "listings_{city}.csv.gz") -> pd.DataFrame:
    """Read ``listings_<city>.csv.gz`` (or .csv) for the given cities and return the typed columns the model needs.

    Only the needed columns are read. Only the cities listed are opened.
    """
    frames = []
    for city in cities:
        path = os.path.join(data_dir, file_pattern.format(city=city))
        if not os.path.exists(path) and path.endswith(".gz"):
            path = path[:-3]
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found")
        header = pd.read_csv(path, nrows=0).columns
        raw = pd.read_csv(path, usecols=[c for c in _RAW_COLUMNS if c in header], low_memory=False)
        raw["city"] = city
        frames.append(_type_raw(raw))
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- features
def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def _to_float(s: pd.Series) -> pd.Series:
    """Numbers, booleans, None and NaN to float64 (Spark's toPandas returns mixed object columns)."""
    if pd.api.types.is_float_dtype(s) or pd.api.types.is_integer_dtype(s) or pd.api.types.is_bool_dtype(s):
        return s.astype("float64")
    return s.map(_num).astype("float64")


def ensure_columns(df: pd.DataFrame):
    """Add required columns that are absent (as missing values) and report them. ``city`` and ``id`` are mandatory."""
    out = df.copy()
    for c in ("city", "id"):
        if c not in out.columns:
            raise ValueError(f"column '{c}' is required")
    missing = [c for c in REQUIRED_COLUMNS if c not in out.columns]
    for c in missing:
        out[c] = None if c in _TEXT_COLUMNS else np.nan
    return out, missing


def add_features(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Return a copy of ``df`` with the numeric columns as floats and every model feature added.

    Uses no price information except for the target column itself, so it can be applied to any city.
    """
    d, missing = ensure_columns(df)
    if missing and verbose:
        print(f"WARNING: columns missing in the input and filled with missing values: {missing}")
    for c in _NUMERIC_RAW:
        d[c] = _to_float(d[c])

    d["minimum_nights_capped"] = d["minimum_nights"].clip(upper=365)
    d["is_monthly_rental"] = np.where(d["minimum_nights"].isna(), np.nan, (d["minimum_nights"] >= 28).astype(float))
    d["stay_type"] = np.where(d["minimum_nights"].isna(), "unknown", np.where(d["minimum_nights"] >= 28, "monthly", "short"))
    d["log_host_listings_count"] = np.log1p(d["host_listings_count"].clip(lower=0))
    d["log_calc_host_listings_count"] = np.log1p(d["calculated_host_listings_count"].clip(lower=0))

    # location relative to the city: the centre is the median point of the city's own listings
    lat_c = d.groupby("city")["latitude"].transform("median")
    lon_c = d.groupby("city")["longitude"].transform("median")
    d["dy_km"] = (d["latitude"] - lat_c) * 111.0
    d["dx_km"] = (d["longitude"] - lon_c) * 111.0 * np.cos(np.radians(lat_c))
    d["dist_to_center_km"] = np.hypot(d["dx_km"], d["dy_km"])
    return d


# --------------------------------------------------------------------------- price validity and split
def valid_price_mask(d: pd.DataFrame) -> pd.Series:
    """Priced and not a placeholder (a number typed to block a listing). Only these rows are used for fitting and scoring."""
    p = d[TARGET]
    per_guest = p / d["accommodates"].where(d["accommodates"] > 0)
    return p.notna() & (p > 0) & ~(per_guest >= PLACEHOLDER_PRICE_PER_GUEST) & ~(p >= PLACEHOLDER_PRICE)


def price_status(d: pd.DataFrame) -> pd.Series:
    """valid / no_price / placeholder for every listing."""
    p = d[TARGET]
    return pd.Series(np.select([p.isna() | (p <= 0), valid_price_mask(d)], ["no_price", "valid"], default="placeholder"), index=d.index)


def make_split(d: pd.DataFrame, test_size: float = 0.2, seed: int = SEED, min_stratum: int = 10) -> pd.Series:
    """'train' / 'test' for listings with a valid price, 'not_labelled' for the rest.

    Stratified by city x room type (rare strata are merged per city), deterministic for a fixed seed.
    """
    split = pd.Series("not_labelled", index=d.index, dtype="object")
    ok = d[valid_price_mask(d)].sort_values(["city", "id"])
    strata = ok["city"].astype(str) + "|" + ok["room_type"].fillna("unknown").astype(str)
    counts = strata.map(strata.value_counts())
    strata = strata.where(counts >= min_stratum, ok["city"].astype(str) + "|other")
    stratify = strata.to_numpy() if strata.value_counts().min() >= 2 else None
    train_idx, test_idx = train_test_split(ok.index.to_numpy(), test_size=test_size, stratify=stratify, random_state=seed)
    split.loc[train_idx] = "train"
    split.loc[test_idx] = "test"
    return split


# --------------------------------------------------------------------------- baseline
class MedianBaseline:
    """Median price of the training listings with the same ``keys`` (default: room type and guests).

    Falls back to the median of the first key (room type), then to the overall median, for combinations never seen in training.
    """

    def __init__(self, keys: Sequence[str] = ("room_type", "accommodates")):
        self.keys = list(keys)

    def fit(self, d: pd.DataFrame) -> "MedianBaseline":
        d = d[valid_price_mask(d)]
        self.table_ = d.groupby(self.keys)[TARGET].agg(median_price="median", n_listings="size")
        self.by_first_ = d.groupby(self.keys[0])[TARGET].median()
        self.global_ = float(d[TARGET].median())
        return self

    def predict(self, d: pd.DataFrame) -> np.ndarray:
        idx = pd.MultiIndex.from_frame(d[self.keys]) if len(self.keys) > 1 else pd.Index(d[self.keys[0]])
        p = pd.Series(self.table_["median_price"].reindex(idx).to_numpy(), index=d.index)
        p = p.fillna(d[self.keys[0]].map(self.by_first_)).fillna(self.global_)
        return p.to_numpy()


# --------------------------------------------------------------------------- model
CANDIDATES = {
    "ridge, alpha 1": ("ridge", {"alpha": 1.0}),
    "ridge, alpha 10": ("ridge", {"alpha": 10.0}),
    "boosting, depth 4": ("hgb", {"max_depth": 4, "learning_rate": 0.08, "max_iter": 400, "min_samples_leaf": 30}),
    "boosting, depth 6": ("hgb", {"max_depth": 6, "learning_rate": 0.06, "max_iter": 400, "min_samples_leaf": 30}),
    "boosting, 31 leaves": ("hgb", {"max_depth": None, "max_leaf_nodes": 31, "learning_rate": 0.05, "max_iter": 500,
                                    "min_samples_leaf": 20, "l2_regularization": 1.0}),
}
_HGB_DEFAULTS = {"learning_rate": 0.06, "max_iter": 400, "max_depth": 6, "min_samples_leaf": 30, "l2_regularization": 1.0}


class PriceModel:
    """Predicts log price with a ridge regression or histogram gradient boosting and returns dollars.

    Works on the output of ``add_features``. Unseen categories are handled (ignored / treated as missing),
    missing numbers are handled (imputed for ridge, native for boosting), so the model runs on any city.
    exp(prediction of log price) estimates the median price, which is what minimises the mean absolute error.
    """

    def __init__(self, kind: str = "hgb", params: Optional[dict] = None, seed: int = SEED, top_property_types: int = 25):
        if kind not in ("ridge", "hgb"):
            raise ValueError("kind must be 'ridge' or 'hgb'")
        self.kind, self.params, self.seed, self.top_property_types = kind, dict(params or {}), seed, top_property_types

    def _estimator(self) -> Pipeline:
        if self.kind == "ridge":
            pre = ColumnTransformer([
                ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
                ("num", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler())]), NUM_FEATURES),
            ])
            return Pipeline([("pre", pre), ("model", Ridge(**{"alpha": 1.0, **self.params}))])
        pre = ColumnTransformer([
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan), CAT_FEATURES),
            ("num", "passthrough", NUM_FEATURES),
        ])
        model = HistGradientBoostingRegressor(categorical_features=list(range(len(CAT_FEATURES))), random_state=self.seed,
                                              **{**_HGB_DEFAULTS, **self.params})
        return Pipeline([("pre", pre), ("model", model)])

    def _matrix(self, d: pd.DataFrame, cols: Optional[Sequence[str]] = None) -> pd.DataFrame:
        # a fitted model always scores with the columns it was fitted on, even if FEATURES is edited later (e.g. for v2)
        x = d[list(cols or FEATURES)].copy()
        x["property_type"] = x["property_type"].where(x["property_type"].isin(self.top_types_), "Other")
        return x

    def fit(self, d: pd.DataFrame) -> "PriceModel":
        d = d[valid_price_mask(d)]
        self.top_types_ = d["property_type"].value_counts().head(self.top_property_types).index.tolist()
        self.pipeline_ = self._estimator().fit(self._matrix(d, FEATURES), np.log(d[TARGET].to_numpy()))
        self.n_train_ = len(d)
        return self

    def predict(self, d: pd.DataFrame) -> np.ndarray:
        return np.exp(self.pipeline_.predict(self._matrix(d, list(getattr(self.pipeline_, "feature_names_in_", FEATURES)))))

    def describe(self) -> dict:
        return {"kind": self.kind, "params": self.params, "n_train": self.n_train_, "features": FEATURES, "seed": self.seed}


# --------------------------------------------------------------------------- metrics and validation
def regression_metrics(y_true, y_pred) -> dict:
    """MAE (dollars) and companions. ``r2_log`` is R-squared on log price; ``mape_pct`` is the mean absolute percentage error."""
    y, p = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    ok = ~(np.isnan(y) | np.isnan(p))
    y, p = y[ok], p[ok]
    err = p - y
    ss_res = float(np.sum((np.log(y) - np.log(p)) ** 2))
    ss_tot = float(np.sum((np.log(y) - np.log(y).mean()) ** 2))
    return {"n": int(len(y)), "mae": float(np.mean(np.abs(err))), "median_ae": float(np.median(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err ** 2))), "mape_pct": float(np.mean(np.abs(err) / y) * 100),
            "mean_error": float(np.mean(err)), "r2_log": 1 - ss_res / ss_tot}


def city_scopes(d: pd.DataFrame, roles: Dict[str, str]) -> Dict[str, pd.Series]:
    """{'A: Los_Angeles': mask, 'B: San_Francisco': mask, 'A+B: both': mask} for a mapping city -> role letter."""
    scopes = {f"{role}: {city}": d["city"] == city for city, role in roles.items()}
    scopes[f"{'+'.join(roles.values())}: both"] = pd.Series(True, index=d.index)
    return scopes


def metrics_table(scored: pd.DataFrame, pred_cols: Dict[str, str], scopes: Dict[str, pd.Series], y_col: str = "actual_price") -> pd.DataFrame:
    """One row per (scope, model) with the metrics of ``regression_metrics``."""
    rows = []
    for scope, mask in scopes.items():
        d = scored[mask]
        for name, col in pred_cols.items():
            rows.append({"scope": scope, "model": name, **regression_metrics(d[y_col], d[col])})
    return pd.DataFrame(rows)


def metrics_to_markdown(results: pd.DataFrame, meta: Optional[dict] = None, digits: int = 2) -> str:
    """The output of ``metrics_table`` as a Markdown table (for the README or a results file)."""
    cols = ["scope", "model", "n", "mae", "median_ae", "rmse", "mape_pct", "mean_error", "r2_log"]
    lines = []
    if meta:
        lines += [f"Model: {meta.get('model_name', '')}, chosen candidate: {meta.get('chosen_candidate', '')}, trained at {meta.get('trained_at', '')}", ""]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for _, row in results[cols].iterrows():
        cells = [str(row[c]) if c in ("scope", "model") else (str(int(row[c])) if c == "n" else f"{row[c]:.{digits}f}") for c in cols]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "mae, median_ae, rmse and mean_error are USD per night on the held-out test listings; mape_pct is a percentage; r2_log is R-squared on log price."]
    return "\n".join(lines) + "\n"


def mae_by(scored: pd.DataFrame, by: List[str], pred_cols: Dict[str, str], y_col: str = "actual_price") -> pd.DataFrame:
    """Number of listings and MAE of every model per group."""
    tmp = scored.copy()
    for name, col in pred_cols.items():
        tmp[f"mae_{name}"] = (tmp[col] - tmp[y_col]).abs()
    spec = {f"mae_{name}": (f"mae_{name}", "mean") for name in pred_cols}
    return tmp.groupby(by, observed=True).agg(n=(y_col, "size"), **spec).reset_index()


def cross_validate(make_estimator: Callable[[], object], d: pd.DataFrame, n_splits: int = 5, seed: int = SEED) -> dict:
    """K-fold MAE (dollars) of anything with fit(df) / predict(df). Use on training rows only."""
    maes = []
    for tr, va in KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(d):
        est = make_estimator().fit(d.iloc[tr])
        valid = d.iloc[va]
        maes.append(float(np.mean(np.abs(est.predict(valid) - valid[TARGET].to_numpy()))))
    return {"cv_mae_mean": float(np.mean(maes)), "cv_mae_std": float(np.std(maes)), "folds": n_splits}


def permutation_importance_mae(model: PriceModel, d: pd.DataFrame, groups: Optional[Dict[str, List[str]]] = None,
                               n_repeats: int = 5, seed: int = SEED) -> pd.DataFrame:
    """How much the MAE grows when a group of features is shuffled across listings (rows of ``d`` need a valid price)."""
    groups = groups or FEATURE_GROUPS
    rng = np.random.default_rng(seed)
    y = d[TARGET].to_numpy()
    base = float(np.mean(np.abs(model.predict(d) - y)))
    rows = []
    for name, cols in groups.items():
        increases = []
        for _ in range(n_repeats):
            shuffled = d.copy()
            perm = rng.permutation(len(d))
            for c in cols:
                shuffled[c] = d[c].to_numpy()[perm]
            increases.append(float(np.mean(np.abs(model.predict(shuffled) - y))) - base)
        rows.append({"feature_group": name, "features": ", ".join(cols), "mae_increase_usd": float(np.mean(increases)),
                     "mae_increase_std": float(np.std(increases)), "mae_increase_pct": float(np.mean(increases)) / base * 100})
    return pd.DataFrame(rows).sort_values("mae_increase_usd", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- scoring (gold table layout)
GOLD_COLUMNS = [
    ("city", "string", "City name"),
    ("id", "bigint", "Inside Airbnb listing id"),
    ("split", "string", "train = used to fit the model, test = held out for evaluation, not_labelled = no valid price so predicted only, new_data = city outside the training set"),
    ("price_status", "string", "valid, no_price, or placeholder (an implausible price typed to block the listing)"),
    ("room_type", "string", "Entire home/apt, Private room, Hotel room or Shared room"),
    ("property_type", "string", "Property type as listed"),
    ("neighbourhood_cleansed", "string", "Neighbourhood of the listing"),
    ("accommodates", "double", "Number of guests the listing accommodates"),
    ("bedrooms", "double", "Number of bedrooms, 0 means studio"),
    ("minimum_nights", "double", "Minimum stay in nights set by the host"),
    ("stay_type", "string", "short = minimum stay under 28 nights, monthly = 28 nights or more"),
    ("latitude", "double", "Latitude"),
    ("longitude", "double", "Longitude"),
    ("actual_price", "double", "Advertised price per night in the region currency"),
    ("predicted_price", "double", "Price per night predicted by the model, region currency"),
    ("baseline_price", "double", "Baseline: training median price for the same room type and number of guests"),
    ("baseline_stay_type_price", "double", "Second baseline: training median for the same room type, guests and stay type"),
    ("error_usd", "double", "predicted_price minus actual_price; negative means the model under-prices the listing"),
    ("abs_error_usd", "double", "Absolute error of the model per night"),
    ("pct_error", "double", "error_usd as a percentage of actual_price"),
    ("baseline_abs_error_usd", "double", "Absolute error of the baseline per night"),
    ("pricing_direction", "string", "under-predicted when the model price is below the actual price, otherwise over-predicted"),
    ("model_name", "string", "Name and version of the model that produced the prediction"),
    ("scored_at", "string", "Timestamp of the scoring run"),
]
GOLD_DDL = ", ".join(f"{name} {typ}" for name, typ, _ in GOLD_COLUMNS)
GOLD_COMMENTS = {name: comment for name, _, comment in GOLD_COLUMNS}


def score_listings(d: pd.DataFrame, model: PriceModel, baseline: MedianBaseline, baseline_stay_type: MedianBaseline,
                   split: Optional[pd.Series] = None, split_label: str = "new_data",
                   model_name: str = "price_model_v1", scored_at: str = "") -> pd.DataFrame:
    """Predictions for every listing in ``d`` (output of ``add_features``), in the layout of the gold table.

    Error columns are filled only for listings with a valid price; the others are predicted but not scored.
    """
    valid = valid_price_mask(d)
    actual = d[TARGET]
    pred = pd.Series(model.predict(d), index=d.index)
    base = pd.Series(baseline.predict(d), index=d.index)
    base2 = pd.Series(baseline_stay_type.predict(d), index=d.index)
    err = (pred - actual).where(valid)

    out = pd.DataFrame(index=d.index)
    out["city"] = d["city"]
    out["id"] = d["id"].astype("int64")
    out["split"] = split if split is not None else split_label
    out["price_status"] = price_status(d)
    out["room_type"] = d["room_type"]
    out["property_type"] = d["property_type"]
    out["neighbourhood_cleansed"] = d["neighbourhood_cleansed"] if "neighbourhood_cleansed" in d.columns else None
    out["accommodates"] = d["accommodates"]
    out["bedrooms"] = d["bedrooms"]
    out["minimum_nights"] = d["minimum_nights"]
    out["stay_type"] = d["stay_type"]
    out["latitude"] = d["latitude"]
    out["longitude"] = d["longitude"]
    out["actual_price"] = actual
    out["predicted_price"] = pred.round(2)
    out["baseline_price"] = base.round(2)
    out["baseline_stay_type_price"] = base2.round(2)
    out["error_usd"] = err.round(2)
    out["abs_error_usd"] = err.abs().round(2)
    out["pct_error"] = (100 * err / actual.where(valid)).round(2)
    out["baseline_abs_error_usd"] = (base - actual).abs().where(valid).round(2)
    out["pricing_direction"] = np.where(err.isna(), None, np.where(err > 0, "over-predicted", "under-predicted"))
    out["model_name"] = model_name
    out["scored_at"] = scored_at
    return out[[name for name, _, _ in GOLD_COLUMNS]]


# --------------------------------------------------------------------------- persistence
def save_model(path: str, model: PriceModel, baseline: MedianBaseline, baseline_stay_type: MedianBaseline, meta: Optional[dict] = None) -> None:
    """Save model, both baselines and a small JSON description next to it (``path`` ends with .joblib)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    joblib.dump({"model": model, "baseline": baseline, "baseline_stay_type": baseline_stay_type, "meta": meta or {}}, path)
    with open(os.path.splitext(path)[0] + ".json", "w") as f:
        json.dump({**(meta or {}), **model.describe()}, f, indent=2, default=str)


def load_model(path: str) -> dict:
    """Returns {'model', 'baseline', 'baseline_stay_type', 'meta'}. The module must be importable (src/ on sys.path)."""
    return joblib.load(path)
