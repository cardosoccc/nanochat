"""
Quick exploration script to discover GigaVerbo's actual structure on HuggingFace.

Run this before prepare_pt_en_mix.py to verify:
1. What columns/fields exist in the dataset
2. How subsets are identified (if at all)
3. What the 'label' field distribution looks like
4. Sample sizes and character distributions

Usage:
    python dev/explore_gigaverbo.py [--max-samples 10000]
"""

import argparse
from collections import Counter


def main():
    parser = argparse.ArgumentParser(description="Explore GigaVerbo dataset structure")
    parser.add_argument("--max-samples", type=int, default=10_000,
                        help="Max samples to examine (default: 10,000)")
    args = parser.parse_args()

    from datasets import load_dataset

    print("Loading GigaVerbo in streaming mode...")
    ds = load_dataset("TucanoBR/GigaVerbo", split="train", streaming=True)

    # Examine first batch to discover schema
    first = next(iter(ds))
    print(f"\nDataset columns: {list(first.keys())}")
    print(f"\nFirst sample:")
    for key, value in first.items():
        if isinstance(value, str) and len(value) > 200:
            print(f"  {key}: {value[:200]}...")
        else:
            print(f"  {key}: {value}")

    # Collect statistics
    print(f"\nScanning {args.max_samples:,} samples...")
    label_counts = Counter()
    name_counts = Counter()
    char_lengths = []
    columns_seen = set()

    for i, sample in enumerate(ds):
        if i >= args.max_samples:
            break

        columns_seen.update(sample.keys())

        # Track label distribution
        if "label" in sample:
            label_counts[sample["label"]] += 1

        # Track subset names if available
        if "name" in sample:
            name_counts[sample["name"]] += 1
        elif "source" in sample:
            name_counts[sample["source"]] += 1
        elif "subset" in sample:
            name_counts[sample["subset"]] += 1

        # Track text lengths
        text = sample.get("text", "")
        char_lengths.append(len(text))

        if (i + 1) % 5000 == 0:
            print(f"  Processed {i + 1:,} samples...")

    print(f"\n{'=' * 60}")
    print(f"Results from {min(args.max_samples, i + 1):,} samples:")
    print(f"{'=' * 60}")

    print(f"\nAll columns found: {sorted(columns_seen)}")

    if label_counts:
        print(f"\nLabel distribution:")
        for label, count in sorted(label_counts.items()):
            pct = count / sum(label_counts.values()) * 100
            print(f"  label={label}: {count:,} ({pct:.1f}%)")

    if name_counts:
        print(f"\nSubset/source distribution:")
        for name, count in name_counts.most_common():
            pct = count / sum(name_counts.values()) * 100
            print(f"  {name}: {count:,} ({pct:.1f}%)")

    if char_lengths:
        import statistics
        print(f"\nText length statistics (characters):")
        print(f"  Min:    {min(char_lengths):,}")
        print(f"  Max:    {max(char_lengths):,}")
        print(f"  Mean:   {statistics.mean(char_lengths):,.0f}")
        print(f"  Median: {statistics.median(char_lengths):,.0f}")
        print(f"  Total:  {sum(char_lengths):,}")

    print(f"\n{'=' * 60}")
    print("Use these findings to adjust GIGAVERBO_NON_SYNTHETIC_SUBSETS")
    print("in dev/prepare_pt_en_mix.py if the subset names differ.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
