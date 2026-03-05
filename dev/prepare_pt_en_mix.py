"""
Prepare a mixed English + Brazilian Portuguese pretraining dataset.

This PySpark job:
1. Reads the 9 non-synthetic subsets from TucanoBR/GigaVerbo on HuggingFace
2. Filters to only high-quality samples (label=1 from BERTimbau text filter)
3. Downloads the required number of ClimbMix English shards
4. Mixes both datasets into randomly shuffled parquet shards
5. Produces output in the same format as nanochat expects:
   - ~250M characters per shard
   - row_group_size=1024
   - zstd compression
   - single "text" column

The output can be uploaded to HuggingFace as a standalone bilingual dataset.

Usage:
    # Local mode (for development/testing)
    spark-submit dev/prepare_pt_en_mix.py \
        --pt-ratio 0.3 \
        --target-tokens 10_500_000_000 \
        --output-dir /path/to/output

    # Cluster mode
    spark-submit --master yarn --num-executors 16 \
        dev/prepare_pt_en_mix.py \
        --pt-ratio 0.3 \
        --target-tokens 10_500_000_000 \
        --output-dir /path/to/output

    # Upload to HuggingFace after generation
    spark-submit dev/prepare_pt_en_mix.py \
        --pt-ratio 0.3 \
        --target-tokens 10_500_000_000 \
        --output-dir /path/to/output \
        --upload-to-hf your-username/pt-en-mix-10b-shuffle

Requirements:
    pip install pyspark datasets pyarrow huggingface_hub

Sizing guide (from nanochat scaling laws):
    - Nanochat uses token-to-parameter ratio of ~10.5:1 (Muon optimizer)
    - For 1B params: ~10.5B tokens needed
    - ClimbMix compression ratio: ~4.8 chars/token
    - So ~10.5B tokens ≈ ~50.4B characters ≈ ~202 shards of 250M chars each
    - At 30% Portuguese: ~60 PT shards + ~142 EN shards = ~202 total shards
"""

import argparse
import os
import math
import time
import hashlib
import random

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, LongType

# ─────────────────────────────────────────────────────────────────────────────
# GigaVerbo subset configuration
# ─────────────────────────────────────────────────────────────────────────────

# The 9 non-synthetic subsets of GigaVerbo identified in the Tucano paper
# (arXiv:2411.07854). These are the subsets that contain naturally-occurring
# Portuguese text (not machine-translated or LLM-generated).
#
# The "name" field in the GigaVerbo-Text-Filter dataset maps to these subsets.
# In the main GigaVerbo dataset, they appear as separate configurations/splits.
#
# Mapping from paper names → HuggingFace subset identifiers:
GIGAVERBO_NON_SYNTHETIC_SUBSETS = [
    "monoHPLT",        # HPLT-PT: web crawl data
    "CrawlPT",         # CC-2023: Common Crawl Portuguese snapshot
    "Wikipedia",        # Portuguese Wikipedia
    "CulturaX",         # CulturaX: multilingual web crawl
    "CommonCrawl",      # CCc100 / MC4-PT: Common Crawl derived
    "ROOTS",            # ROOTS Wikiquote + Ted Talks
    "XLSum",            # XL-Sum: news summaries
    "CorpusCarolina",   # Corpus Carolina: reference corpus
    "LegalPT",          # Legal Portuguese: legal documents
]

# Known synthetic subsets to exclude (for reference)
GIGAVERBO_SYNTHETIC_SUBSETS = [
    "InstructPTBR",     # Llama 2-based synthetic
    "UltrachatBR",      # LLM-generated conversations
    "Gpt4all",          # GPT-3.5-Turbo distilled
    "BactrianX",        # GPT-generated multilingual instructions
    "CosmosQA",         # Synthetic QA
]

# ClimbMix English data source (nanochat's current pretraining dataset)
CLIMBMIX_BASE_URL = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"
CLIMBMIX_MAX_SHARD = 6542

# Output format matching nanochat expectations
CHARS_PER_SHARD = 250_000_000  # ~250M characters per shard → ~100MB compressed
ROW_GROUP_SIZE = 1024          # Must be power of 2 for DDP dataloader
CHARS_PER_TOKEN = 4.8          # Approximate compression ratio from nanochat tokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare mixed English + Portuguese pretraining dataset"
    )
    parser.add_argument(
        "--pt-ratio", type=float, default=0.3,
        help="Fraction of the final dataset that should be Portuguese (default: 0.3 = 30%%)"
    )
    parser.add_argument(
        "--target-tokens", type=float, default=10_500_000_000,
        help="Total number of tokens for the final dataset (default: 10.5B for 1B param model)"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Directory to write output shards to"
    )
    parser.add_argument(
        "--quality-filter", action="store_true", default=True,
        help="Filter GigaVerbo to only high-quality samples (label=1). Default: True"
    )
    parser.add_argument(
        "--no-quality-filter", action="store_false", dest="quality_filter",
        help="Disable quality filtering on GigaVerbo"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for shuffling (default: 42)"
    )
    parser.add_argument(
        "--upload-to-hf", type=str, default=None,
        help="HuggingFace repo ID to upload to (e.g., 'username/dataset-name'). Requires HF_TOKEN env var."
    )
    parser.add_argument(
        "--spark-master", type=str, default="local[*]",
        help="Spark master URL (default: local[*] for local mode)"
    )
    parser.add_argument(
        "--gigaverbo-cache-dir", type=str, default=None,
        help="Local cache directory for GigaVerbo data. If not set, uses HF default cache."
    )
    parser.add_argument(
        "--climbmix-cache-dir", type=str, default=None,
        help="Local cache directory for ClimbMix shards. If not set, uses ~/.cache/nanochat/base_data_climbmix/"
    )
    return parser.parse_args()


def compute_sizing(target_tokens: float, pt_ratio: float) -> dict:
    """
    Compute how much data we need from each source.

    Returns a dict with character and shard counts for both languages.
    """
    total_chars = target_tokens * CHARS_PER_TOKEN
    pt_chars = total_chars * pt_ratio
    en_chars = total_chars * (1 - pt_ratio)

    pt_shards = math.ceil(pt_chars / CHARS_PER_SHARD)
    en_shards = math.ceil(en_chars / CHARS_PER_SHARD)
    total_shards = pt_shards + en_shards

    return {
        "target_tokens": target_tokens,
        "total_chars": total_chars,
        "pt_chars": pt_chars,
        "en_chars": en_chars,
        "pt_shards": pt_shards,
        "en_shards": en_shards,
        "total_shards": total_shards,
        "pt_ratio": pt_ratio,
    }


def load_gigaverbo_spark(spark, subsets, quality_filter=True, cache_dir=None):
    """
    Load GigaVerbo non-synthetic subsets using Spark.

    The GigaVerbo dataset on HuggingFace has a 'text' column and a 'label' column.
    label=1 means the BERTimbau text filter classified the sample as high quality.

    We use the HuggingFace datasets library to stream the data and convert to
    a Spark DataFrame for scalable processing.
    """
    from datasets import load_dataset

    print(f"Loading GigaVerbo subsets: {subsets}")
    print(f"Quality filter enabled: {quality_filter}")

    all_texts = []
    total_loaded = 0
    total_filtered = 0

    # Stream through GigaVerbo and collect Portuguese texts
    # GigaVerbo is a single dataset; we need to identify subsets by examining the data
    # The 'name' field (if present) or source identification helps filter subsets
    ds_kwargs = {
        "path": "TucanoBR/GigaVerbo",
        "split": "train",
        "streaming": True,
    }
    if cache_dir:
        ds_kwargs["cache_dir"] = cache_dir

    ds = load_dataset(**ds_kwargs)

    for sample in ds:
        total_loaded += 1

        # Apply quality filter if enabled
        if quality_filter and sample.get("label", 1) == 0:
            total_filtered += 1
            continue

        text = sample.get("text", "")
        if text and len(text.strip()) > 100:  # Skip very short documents
            all_texts.append(text)

        if total_loaded % 1_000_000 == 0:
            print(f"  Processed {total_loaded:,} samples, kept {len(all_texts):,}, "
                  f"filtered {total_filtered:,}")

    print(f"GigaVerbo loading complete: {total_loaded:,} total, "
          f"{len(all_texts):,} kept, {total_filtered:,} filtered")

    # Convert to Spark DataFrame
    schema = StructType([
        StructField("text", StringType(), False),
        StructField("source", StringType(), False),
    ])

    rows = [(text, "gigaverbo_pt") for text in all_texts]
    df = spark.createDataFrame(rows, schema)
    return df


def load_climbmix_spark(spark, num_shards, cache_dir=None):
    """
    Load ClimbMix English shards as a Spark DataFrame.

    Reads the parquet files directly (they already have a 'text' column).
    We randomly select num_shards from the available ClimbMix data.
    """
    import pyarrow.parquet as pq

    if cache_dir is None:
        base_dir = os.path.expanduser("~/.cache/nanochat")
        cache_dir = os.path.join(base_dir, "base_data_climbmix")

    # List available shards
    if os.path.exists(cache_dir):
        available_shards = sorted([
            f for f in os.listdir(cache_dir)
            if f.endswith('.parquet') and not f.endswith('.tmp')
        ])
        # Exclude the last shard (validation set in nanochat convention)
        available_shards = available_shards[:-1] if len(available_shards) > 1 else available_shards
    else:
        available_shards = []

    if len(available_shards) < num_shards:
        print(f"WARNING: Only {len(available_shards)} ClimbMix shards available locally, "
              f"but {num_shards} requested.")
        print(f"Please download more shards first:")
        print(f"  python -m nanochat.dataset -n {num_shards}")
        print(f"Using all {len(available_shards)} available shards.")
        selected_shards = available_shards
    else:
        # Randomly select shards
        rng = random.Random(42)
        selected_shards = rng.sample(available_shards, num_shards)

    print(f"Loading {len(selected_shards)} ClimbMix shards from {cache_dir}")

    # Read parquet files with Spark
    shard_paths = [os.path.join(cache_dir, s) for s in selected_shards]

    if not shard_paths:
        print("ERROR: No ClimbMix shards available. Download them first:")
        print(f"  python -m nanochat.dataset -n {num_shards}")
        raise FileNotFoundError("No ClimbMix shards found")

    df = spark.read.parquet(*shard_paths)
    df = df.select(
        F.col("text"),
        F.lit("climbmix_en").alias("source"),
    )

    return df


def write_mixed_shards(df, output_dir, seed=42):
    """
    Write the mixed DataFrame into nanochat-compatible parquet shards.

    The output format matches exactly what nanochat's dataloader expects:
    - Single 'text' column
    - ~250M characters per shard
    - row_group_size=1024
    - zstd compression at level 3
    - Files named shard_XXXXX.parquet
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    os.makedirs(output_dir, exist_ok=True)

    print(f"Writing shuffled shards to {output_dir}")
    print(f"Target: ~{CHARS_PER_SHARD:,} chars/shard, row_group_size={ROW_GROUP_SIZE}")

    # Collect all texts (shuffled by Spark)
    # We process in batches for memory efficiency
    texts = df.select("text").rdd.flatMap(lambda row: [row.text]).collect()

    # Shuffle with seed for reproducibility
    rng = random.Random(seed)
    rng.shuffle(texts)

    print(f"Total documents to write: {len(texts):,}")

    # Write shards in the same format as repackage_data_reference.py
    shard_docs = []
    shard_index = 0
    shard_characters = 0
    total_docs_written = 0
    total_chars_written = 0
    t0 = time.time()

    for text in texts:
        shard_docs.append(text)
        shard_characters += len(text)

        collected_enough_chars = shard_characters >= CHARS_PER_SHARD
        docs_multiple_of_row_group_size = len(shard_docs) % ROW_GROUP_SIZE == 0

        if collected_enough_chars and docs_multiple_of_row_group_size:
            shard_path = os.path.join(output_dir, f"shard_{shard_index:05d}.parquet")
            shard_table = pa.Table.from_pydict({"text": shard_docs})
            pq.write_table(
                shard_table,
                shard_path,
                row_group_size=ROW_GROUP_SIZE,
                use_dictionary=False,
                compression="zstd",
                compression_level=3,
                write_statistics=False,
            )

            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            total_docs_written += len(shard_docs)
            total_chars_written += shard_characters
            remaining = len(texts) - total_docs_written
            print(f"Wrote {shard_path} | docs: {len(shard_docs):,} | "
                  f"chars: {shard_characters:,} | time: {dt:.1f}s | "
                  f"remaining docs: {remaining:,}")

            shard_docs = []
            shard_characters = 0
            shard_index += 1

    # Write any remaining documents as the last shard (validation set)
    if shard_docs:
        # Pad to row_group_size boundary if needed
        remainder = len(shard_docs) % ROW_GROUP_SIZE
        if remainder > 0:
            # Trim to last complete row group
            shard_docs = shard_docs[:len(shard_docs) - remainder]

        if shard_docs:
            shard_path = os.path.join(output_dir, f"shard_{shard_index:05d}.parquet")
            shard_table = pa.Table.from_pydict({"text": shard_docs})
            pq.write_table(
                shard_table,
                shard_path,
                row_group_size=ROW_GROUP_SIZE,
                use_dictionary=False,
                compression="zstd",
                compression_level=3,
                write_statistics=False,
            )
            total_docs_written += len(shard_docs)
            total_chars_written += shard_characters
            shard_index += 1
            print(f"Wrote final shard {shard_path} (validation) | docs: {len(shard_docs):,}")

    print(f"\nDone! Wrote {shard_index} shards")
    print(f"Total documents: {total_docs_written:,}")
    print(f"Total characters: {total_chars_written:,}")
    print(f"Estimated tokens: {total_chars_written / CHARS_PER_TOKEN:,.0f}")

    return shard_index


def upload_to_huggingface(output_dir, repo_id):
    """Upload the output shards to HuggingFace."""
    from huggingface_hub import HfApi

    token = os.getenv("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN environment variable not set. Cannot upload.")
        return

    print(f"Uploading to HuggingFace: {repo_id}")
    api = HfApi(token=token)
    api.upload_large_folder(
        folder_path=output_dir,
        repo_id=repo_id,
        repo_type="dataset",
    )
    print(f"Upload complete: https://huggingface.co/datasets/{repo_id}")


def main():
    args = parse_args()

    # ─── Compute sizing ───────────────────────────────────────────────────
    sizing = compute_sizing(args.target_tokens, args.pt_ratio)

    print("=" * 70)
    print("  Portuguese + English Pretraining Data Preparation")
    print("=" * 70)
    print()
    print(f"  Target tokens:      {sizing['target_tokens']:>15,.0f}")
    print(f"  Total characters:   {sizing['total_chars']:>15,.0f}")
    print(f"  Portuguese ratio:   {sizing['pt_ratio']:>15.1%}")
    print(f"  PT characters:      {sizing['pt_chars']:>15,.0f}")
    print(f"  EN characters:      {sizing['en_chars']:>15,.0f}")
    print(f"  PT shards needed:   {sizing['pt_shards']:>15,}")
    print(f"  EN shards needed:   {sizing['en_shards']:>15,}")
    print(f"  Total shards:       {sizing['total_shards']:>15,}")
    print(f"  Output directory:   {args.output_dir}")
    print()
    print("=" * 70)

    # ─── Initialize Spark ─────────────────────────────────────────────────
    spark = (
        SparkSession.builder
        .appName("nanochat-pt-en-mix")
        .master(args.spark_master)
        .config("spark.driver.memory", "8g")
        .config("spark.sql.parquet.compression.codec", "zstd")
        .getOrCreate()
    )

    try:
        # ─── Load Portuguese data from GigaVerbo ──────────────────────────
        print("\n[1/4] Loading Portuguese data from GigaVerbo...")
        pt_df = load_gigaverbo_spark(
            spark,
            subsets=GIGAVERBO_NON_SYNTHETIC_SUBSETS,
            quality_filter=args.quality_filter,
            cache_dir=args.gigaverbo_cache_dir,
        )

        # Limit to target Portuguese characters
        # We add a character count column and use cumulative sum to cut off
        pt_df = pt_df.withColumn("char_count", F.length("text"))
        pt_total_chars = pt_df.agg(F.sum("char_count")).collect()[0][0]
        print(f"  Portuguese data available: {pt_total_chars:,} characters")

        if pt_total_chars < sizing["pt_chars"]:
            print(f"  WARNING: Not enough Portuguese data ({pt_total_chars:,} < {sizing['pt_chars']:,.0f})")
            print(f"  Will use all available Portuguese data and adjust EN ratio.")
            actual_pt_chars = pt_total_chars
            # Recalculate EN to maintain total
            actual_en_chars = sizing["total_chars"] - actual_pt_chars
        else:
            actual_pt_chars = sizing["pt_chars"]
            actual_en_chars = sizing["en_chars"]
            # Sample down to target size
            sample_ratio = sizing["pt_chars"] / pt_total_chars
            if sample_ratio < 1.0:
                pt_df = pt_df.sample(fraction=sample_ratio, seed=args.seed)
                print(f"  Sampled {sample_ratio:.1%} of Portuguese data")

        pt_df = pt_df.select("text", "source")

        # ─── Load English data from ClimbMix ──────────────────────────────
        en_shards_needed = math.ceil(actual_en_chars / CHARS_PER_SHARD)
        print(f"\n[2/4] Loading English data from ClimbMix ({en_shards_needed} shards)...")
        en_df = load_climbmix_spark(
            spark,
            num_shards=en_shards_needed,
            cache_dir=args.climbmix_cache_dir,
        )

        # ─── Mix and shuffle ──────────────────────────────────────────────
        print("\n[3/4] Mixing and shuffling datasets...")
        mixed_df = pt_df.union(en_df)

        # Add a random sort key for shuffling
        mixed_df = mixed_df.withColumn(
            "sort_key",
            F.rand(seed=args.seed)
        ).orderBy("sort_key").drop("sort_key", "source")

        # ─── Write output shards ──────────────────────────────────────────
        print(f"\n[4/4] Writing output shards to {args.output_dir}...")
        num_shards = write_mixed_shards(mixed_df, args.output_dir, seed=args.seed)

        # ─── Summary ──────────────────────────────────────────────────────
        print()
        print("=" * 70)
        print("  Dataset Preparation Complete!")
        print("=" * 70)
        print(f"  Output:  {args.output_dir}")
        print(f"  Shards:  {num_shards}")
        print(f"  Format:  parquet (zstd, ~250M chars/shard, row_group=1024)")
        print()
        print("  To use with nanochat:")
        print(f"    1. Upload to HuggingFace or copy to ~/.cache/nanochat/base_data_custom/")
        print(f"    2. Update nanochat/dataset.py BASE_URL and DATA_DIR")
        print(f"    3. Retrain tokenizer: python -m scripts.tok_train")
        print(f"    4. Train: torchrun --nproc_per_node=8 -m scripts.base_train --depth=24")
        print("=" * 70)

        # ─── Optional HuggingFace upload ──────────────────────────────────
        if args.upload_to_hf:
            upload_to_huggingface(args.output_dir, args.upload_to_hf)

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
