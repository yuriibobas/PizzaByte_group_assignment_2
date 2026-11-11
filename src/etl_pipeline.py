"""City-parameterised Inside Airbnb ingestion (assignment 2, section 3.3).

Only the selected archive is opened. Raw archives are immutable and content-
addressed; each Delta refresh replaces one city, never the whole dataset.
The module neither trains a model nor writes analytical/prediction gold tables.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
import shutil

from pyspark.sql import DataFrame, Window, functions as F


CITY_FILES = {
    "Los_Angeles": "listings_Los_Angeles.csv.gz",
    "San_Francisco": "listings_San_Francisco.csv.gz",
    "Santa_Clara_County": "listings_Santa_Clara_County.csv.gz",
}
TRAINING_CITIES = ("Los_Angeles", "San_Francisco")
EXPECTED_ROWS = {"Los_Angeles": 43_781, "San_Francisco": 7_405}
NULL_TEXT = ("", "n/a", "na", "nan", "null", "none", "-", "?")

ID_COLS = ("id", "scrape_id", "host_id")
INT_COLS = (
    "host_listings_count", "calculated_host_listings_count",
    "calculated_host_listings_count_entire_homes", "calculated_host_listings_count_private_rooms",
    "calculated_host_listings_count_shared_rooms", "hosts_time_as_host_years", "accommodates",
    "minimum_nights", "maximum_nights", "availability_30", "availability_60", "availability_90",
    "availability_365", "availability_eoy", "number_of_reviews", "number_of_reviews_ltm",
    "number_of_reviews_l30d", "estimated_occupancy_l365d",
)
DOUBLE_COLS = (
    "latitude", "longitude", "bedrooms", "beds", "price_quote_total_price",
    "price_quote_price_per_night", "estimated_revenue_l365d", "reviews_per_month",
    "review_scores_rating", "review_scores_accuracy", "review_scores_cleanliness",
    "review_scores_checkin", "review_scores_communication", "review_scores_location", "review_scores_value",
)
DATE_COLS = ("last_scraped", "first_review", "last_review", "price_quote_checkin_date", "price_quote_checkout_date")
BOOL_COLS = ("host_is_superhost", "host_identity_verified", "host_has_profile_pic", "has_availability")
STR_COLS = (
    "city", "source", "name", "description", "property_type", "room_type", "neighbourhood_cleansed",
    "neighbourhood_group_cleansed", "bathrooms_text", "amenities", "license", "price_quote_raw",
)
LINEAGE_COLS = ("_source_file", "_source_sha256", "_ingested_at", "_record_hash")

# Exact raw input contract in src/price_model.py, including target and keys.
# Feature engineering/fitting stays with the model, not this projection.
MODEL_INPUT_COLUMNS = (
    "city", "id", "room_type", "property_type", "price", "accommodates", "bedrooms", "beds",
    "bathrooms_num", "bathroom_shared", "n_amenities", "minimum_nights", "host_is_superhost",
    "host_listings_count", "calculated_host_listings_count", "hosts_time_as_host_years",
    "has_license", "latitude", "longitude",
)
MODEL_METADATA_COLUMNS = ("neighbourhood_cleansed", "last_scraped", "price_valid", "price_class", "stay_type", "is_stale")


@dataclass(frozen=True)
class PipelineConfig:
    city: str = "Los_Angeles"
    catalog: str = "workspace"
    schema: str = "airbnb"
    source_dir: str = ""
    allow_city_c: bool = False

    def validate(self) -> None:
        if self.city not in CITY_FILES:
            raise ValueError(f"Unknown city {self.city!r}; expected one of {list(CITY_FILES)}")
        if self.city == "Santa_Clara_County" and self.allow_city_c is not True:
            raise ValueError("City C is blocked. Freeze the A/B pipeline and model before explicitly enabling allow_city_c.")
        for label, value in (("catalog", self.catalog), ("schema", self.schema)):
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
                raise ValueError(f"Invalid {label}: {value!r}")

    @property
    def namespace(self) -> str:
        return f"{self.catalog}.{self.schema}"

    @property
    def volume_path(self) -> Path:
        return Path(f"/Volumes/{self.catalog}/{self.schema}/raw")

    @property
    def tables(self) -> dict[str, str]:
        return {layer: f"{self.namespace}.{name}" for layer, name in (
            ("bronze", "bronze_listings"), ("silver", "silver_listings"), ("features", "gold_model_features"))}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stage_archive(spark, config: PipelineConfig) -> tuple[Path, str]:
    config.validate()  # Must precede even looking for C's archive.
    source = Path(config.source_dir) / CITY_FILES[config.city]
    if not source.is_file():
        raise FileNotFoundError(f"Selected archive not found: {source}")
    fingerprint = file_sha256(source)
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {config.namespace}")
    spark.sql(f"CREATE VOLUME IF NOT EXISTS {config.namespace}.raw")
    target = config.volume_path / config.city / fingerprint / source.name
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if file_sha256(target) != fingerprint:
            raise ValueError(f"Staged archive checksum mismatch: {target}")
    else:
        shutil.copyfile(source, target)
        if file_sha256(target) != fingerprint:
            raise ValueError(f"Archive copy checksum mismatch: {target}")
    return target, fingerprint


def read_bronze(spark, path: str, city: str, fingerprint: str) -> DataFrame:
    raw = (spark.read.option("header", True).option("inferSchema", False)
           .option("multiLine", True).option("quote", '"').option("escape", '"')
           .option("mode", "FAILFAST").csv(path))
    if "id" not in raw.columns:
        raise ValueError("The archive is missing the required listing id column")
    reserved = set(raw.columns) & set(LINEAGE_COLS)
    if reserved:
        raise ValueError(f"Source contains reserved lineage columns: {sorted(reserved)}")
    record = F.struct(*[F.col(f"`{name}`") for name in sorted(raw.columns)])
    return (raw.withColumn("city", F.lit(city))
            .withColumn("_source_file", F.lit(path))
            .withColumn("_source_sha256", F.lit(fingerprint))
            .withColumn("_ingested_at", F.current_timestamp())
            .withColumn("_record_hash", F.sha2(F.to_json(record), 256)))


def _finite(column):
    return column.isNotNull() & ~F.isnan(column) & (F.abs(column) != F.lit(float("inf")))


def transform_silver(raw: DataFrame) -> DataFrame:
    """EDA-compatible types and deterministic, non-fitted derived fields.

    price_class describes fixed-rule label eligibility (valid/no_price/placeholder).
    Statistical high/low outlier diagnostics remain in EDA and never filter test rows.
    """
    needed = set(ID_COLS + INT_COLS + DOUBLE_COLS + DATE_COLS + BOOL_COLS + STR_COLS + ("price",))
    columns = []
    for name in sorted(needed):
        if name not in raw.columns:
            value = F.lit(None).cast("string")
        elif name != "city":
            value = F.col(name).cast("string")
            value = F.when(F.lower(F.trim(value)).isin(*NULL_TEXT), None).otherwise(value)
        else:
            value = F.col(name)
        columns.append(value.alias(name))
    cleaned = raw.select(*columns, *[F.col(c) for c in LINEAGE_COLS])

    select = [F.expr(f"try_cast(`{c}` AS BIGINT)").alias(c) for c in ID_COLS]
    select += [F.expr(f"try_cast(try_cast(`{c}` AS DOUBLE) AS BIGINT)").alias(c) for c in INT_COLS]
    for name in DOUBLE_COLS:
        value = F.expr(f"try_cast(`{name}` AS DOUBLE)")
        select.append(F.when(_finite(value), value).alias(name))
    select += [F.expr(f"try_cast(`{c}` AS DATE)").alias(c) for c in DATE_COLS]
    select += [F.when(F.lower(F.trim(F.col(c))) == "t", True)
               .when(F.lower(F.trim(F.col(c))) == "f", False).cast("boolean").alias(c) for c in BOOL_COLS]
    select += [F.col(c).cast("string").alias(c) for c in STR_COLS]
    price = F.expr("try_cast(regexp_replace(price, '[$,]', '') AS DOUBLE)")
    select += [F.col("price").alias("price_text"), F.when(_finite(price), price).alias("price")]
    select += [F.col(c) for c in LINEAGE_COLS]
    typed = cleaned.select(*select)

    bath = F.lower(F.col("bathrooms_text"))
    digits = F.regexp_extract("bathrooms_text", r"(\d+(?:\.\d+)?)", 1)
    amenities = F.from_json("amenities", "array<string>")
    typed = (typed
        .withColumn("bathrooms_num", F.when(bath.isNull(), F.lit(None).cast("double"))
            .when(bath.contains("half") & (digits == ""), F.lit(0.5))
            .when(digits == "", F.lit(None).cast("double"))
            .otherwise(digits.cast("double")))
        .withColumn("bathroom_shared", F.when(bath.isNotNull(), bath.contains("shared")))
        .withColumn("n_amenities", F.when(amenities.isNotNull(), F.size(amenities)))
        .withColumn("amenities_parse_failed", F.col("amenities").isNotNull() & amenities.isNull())
        .withColumn("has_license", F.col("license").isNotNull())
        .withColumn("is_stale", F.coalesce(F.col("source") == "previous scrape", F.lit(False)))
        .withColumn("source_label", F.when(F.col("is_stale"), "previous scrape").otherwise("city scrape"))
        .withColumn("is_monthly_rental", F.col("minimum_nights") >= 28)
        .withColumn("stay_type", F.when(F.col("minimum_nights").isNull(), "unknown")
                    .when(F.col("minimum_nights") >= 28, "monthly").otherwise("short"))
        # Original minimum_nights stays unchanged: the model already caps it.
        .withColumn("minimum_nights_capped", F.least(F.col("minimum_nights"), F.lit(365)))
        .withColumn("minimum_nights_capped", F.when(F.col("minimum_nights").isNotNull(), F.col("minimum_nights_capped")))
        .withColumn("maximum_nights_raw", F.col("maximum_nights"))
        .withColumn("maximum_nights", F.when(F.col("maximum_nights") < 1125, F.col("maximum_nights")))
        .withColumn("price_per_guest", F.expr("try_divide(price, accommodates)"))
        .withColumn("quote_is_available", F.get_json_object("price_quote_raw", "$.quote.is_available") == "true")
        .withColumn("quote_is_monthly", F.get_json_object("price_quote_raw", "$.quote.is_monthly_priced") == "true")
        .withColumn("quote_json_price_per_night", F.expr("try_cast(get_json_object(price_quote_raw, '$.quote.price_per_night') AS DOUBLE)"))
        .withColumn("quote_nights", F.datediff("price_quote_checkout_date", "price_quote_checkin_date"))
        .withColumn("quote_lead_days", F.datediff("price_quote_checkin_date", "last_scraped")))

    # Same fixed eligibility rule as price_model.valid_price_mask, including
    # its treatment of a missing/nonpositive guest count.
    placeholder = F.coalesce(F.col("price") >= 99999, F.lit(False)) | F.coalesce(
        (F.col("accommodates") > 0) & (F.col("price_per_guest") >= 2500), F.lit(False))
    valid = F.coalesce((F.col("price") > 0) & ~placeholder, F.lit(False))
    typed = (typed.withColumn("price_valid", valid)
        .withColumn("price_class", F.when(F.col("price").isNull() | (F.col("price") <= 0), "no_price")
                    .when(placeholder, "placeholder").otherwise("valid"))
        .withColumn("price_status", F.col("price_class")))

    if typed.filter(F.col("id").isNull()).limit(1).count():
        raise ValueError("A listing id is missing or cannot be parsed as BIGINT; silver publication refused")
    order = Window.partitionBy("city", "id").orderBy(F.col("last_scraped").desc_nulls_last(), F.col("_record_hash").desc())
    return typed.withColumn("_key_rank", F.row_number().over(order)).filter("_key_rank = 1").drop("_key_rank")


def model_feature_projection(silver: DataFrame) -> DataFrame:
    return silver.select(*(MODEL_INPUT_COLUMNS + MODEL_METADATA_COLUMNS))


def validate_silver(silver: DataFrame, expected_city: str) -> dict:
    if silver.filter(F.col("city").isNull() | (F.col("city") != expected_city)).limit(1).count():
        raise AssertionError("Unexpected city in selected-city output")
    missing = set(MODEL_INPUT_COLUMNS) - set(silver.columns)
    if missing:
        raise AssertionError(f"Missing model input columns: {sorted(missing)}")
    if dict(silver.dtypes)["id"] != "bigint" or dict(silver.dtypes)["price"] != "double":
        raise AssertionError("id must be BIGINT and price must be DOUBLE")
    result = silver.agg(
        F.count("*").alias("rows"), F.countDistinct("id").alias("distinct_ids"),
        F.sum(F.col("price_valid").cast("long")).alias("valid_prices"),
        F.sum((F.col("price_class") == "no_price").cast("long")).alias("no_price"),
        F.sum((F.col("price_class") == "placeholder").cast("long")).alias("placeholders"),
        F.sum(F.col("is_stale").cast("long")).alias("stale"),
        F.sum(F.col("amenities_parse_failed").cast("long")).alias("amenities_parse_failures"),
        F.min("last_scraped").alias("first_scraped"), F.max("last_scraped").alias("last_scraped"),
    ).first().asDict()
    if not result["rows"] or result["rows"] != result["distinct_ids"]:
        raise AssertionError("Silver must have a nonempty unique listing key")
    if result["valid_prices"] + result["no_price"] + result["placeholders"] != result["rows"]:
        raise AssertionError("Price eligibility classes do not cover all listings")
    return result


def _other_ab_signature(spark, table_name: str, city: str):
    if not spark.catalog.tableExists(table_name):
        return []
    df = spark.table(table_name).filter(F.col("city").isin(*TRAINING_CITIES) & (F.col("city") != city))
    # No C records are requested, even if they exist after the later experiment.
    return sorted((r["city"], r["n"], r["ids"]) for r in df.groupBy("city").agg(
        F.count("*").alias("n"), F.countDistinct("id").alias("ids")).collect())


def write_city(spark, df: DataFrame, table_name: str, city: str) -> None:
    if city not in CITY_FILES:
        raise ValueError(f"Unknown city: {city!r}")
    if df.filter(F.col("city").isNull() | (F.col("city") != city)).limit(1).count():
        raise ValueError("Refusing a city refresh containing another city")
    before = _other_ab_signature(spark, table_name, city)
    writer = df.write.format("delta").mode("overwrite")
    if spark.catalog.tableExists(table_name):
        if spark.sql(f"DESCRIBE DETAIL {table_name}").select("format").first()[0] != "delta":
            raise ValueError(f"Refusing to replace a non-Delta table: {table_name}")
        writer.option("replaceWhere", f"city = '{city}'").option("mergeSchema", "true").saveAsTable(table_name)
    else:
        writer.partitionBy("city").saveAsTable(table_name)
    after = _other_ab_signature(spark, table_name, city)
    if before != after:
        raise AssertionError(f"Another A/B city's counts changed in {table_name}")


def table_signature(spark, table_name: str, cities=TRAINING_CITIES) -> list:
    """Stable content checksums for explicitly requested cities (no default C read)."""
    df = spark.table(table_name).filter(F.col("city").isin(list(cities)))
    columns = sorted(c for c in df.columns if c != "_ingested_at")
    signature = df.groupBy("city").agg(
        F.count("*").alias("rows"), F.countDistinct("id").alias("ids"),
        F.sum(F.xxhash64(*columns).cast("decimal(38,0)")).alias("content_hash"))
    return sorted((r["city"], r["rows"], r["ids"], str(r["content_hash"])) for r in signature.collect())


def run_pipeline(spark, config: PipelineConfig) -> dict:
    config.validate()
    staged, fingerprint = stage_archive(spark, config)
    raw = read_bronze(spark, str(staged), config.city, fingerprint)
    raw_count = raw.count()
    if not raw_count:
        raise ValueError("The selected archive contains no listing records")
    if config.city in EXPECTED_ROWS and raw_count != EXPECTED_ROWS[config.city]:
        raise AssertionError(f"{config.city}: expected {EXPECTED_ROWS[config.city]:,} supplied-file rows, got {raw_count:,}")
    write_city(spark, raw, config.tables["bronze"], config.city)
    # Durable bronze/silver materialisation avoids depending on cache support.
    staged_raw = spark.table(config.tables["bronze"]).filter(F.col("city") == config.city)
    silver = transform_silver(staged_raw)
    summary = validate_silver(silver, config.city)
    write_city(spark, silver, config.tables["silver"], config.city)
    published_silver = spark.table(config.tables["silver"]).filter(F.col("city") == config.city)
    write_city(spark, model_feature_projection(published_silver), config.tables["features"], config.city)
    for layer, table_name in config.tables.items():
        expected = raw_count if layer == "bronze" else summary["rows"]
        actual = spark.table(table_name).filter(F.col("city") == config.city).count()
        if actual != expected:
            raise AssertionError(f"Published row count mismatch in {table_name}: {actual} != {expected}")
    return {"city": config.city, "source_sha256": fingerprint, "staged_file": str(staged),
            "bronze_rows": raw_count, "duplicate_rows_removed": raw_count - summary["rows"],
            **summary, "tables": config.tables}
