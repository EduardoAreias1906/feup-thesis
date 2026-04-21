"""
=============================================================================
inspect_data.py — Quick Data Inspector for VS Code Terminal
=============================================================================
Run after generate_synthetic_data.py to browse your tables.

Usage:
  python inspect_data.py                    # interactive menu
  python inspect_data.py --file students    # jump straight to a table
  python inspect_data.py --all              # print summary of all tables

Requirements: pandas (already installed for the generator)
=============================================================================
"""

import argparse
import os
import pandas as pd

DATA_DIR = "./data"

FILES = {
    "students":          "students.csv",
    "activity_logs":     "activity_logs.csv",
    "interactions":      "interactions.csv",
    "academic_outcomes": "academic_outcomes.csv",
}


def load_table(name: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, FILES[name])
    if not os.path.exists(path):
        print(f"  ✗ File not found: {path}")
        print(f"    Run generate_synthetic_data.py first.")
        return None
    return pd.read_csv(path)


def show_overview(df: pd.DataFrame, name: str):
    """Print shape, dtypes, and first rows."""
    print(f"\n{'=' * 60}")
    print(f"  {name.upper()}  —  {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"{'=' * 60}")
    print(f"\nColumns and types:")
    print(f"  {'Column':<30} {'Type':<15} {'Non-Null':>10} {'Unique':>10}")
    print(f"  {'-'*30} {'-'*15} {'-'*10} {'-'*10}")
    for col in df.columns:
        dtype = str(df[col].dtype)
        non_null = df[col].notna().sum()
        unique = df[col].nunique()
        print(f"  {col:<30} {dtype:<15} {non_null:>10,} {unique:>10,}")


def show_head(df: pd.DataFrame, n: int = 10):
    """Show first n rows with all columns visible."""
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", 40)
    print(f"\nFirst {n} rows:")
    print(df.head(n).to_string(index=False))


def show_stats(df: pd.DataFrame):
    """Numeric summary stats."""
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        print("\n  No numeric columns.")
        return
    pd.set_option("display.float_format", lambda x: f"{x:.2f}")
    print(f"\nNumeric statistics:")
    print(numeric.describe().to_string())


def show_categoricals(df: pd.DataFrame, max_unique: int = 20):
    """Value counts for low-cardinality columns."""
    print(f"\nCategorical value counts (columns with ≤{max_unique} unique values):")
    for col in df.columns:
        if df[col].nunique() <= max_unique and df[col].dtype == "object":
            print(f"\n  {col}:")
            counts = df[col].value_counts()
            for val, cnt in counts.items():
                pct = cnt / len(df) * 100
                bar = "█" * int(pct / 2)
                print(f"    {val:<30} {cnt:>6,}  ({pct:5.1f}%) {bar}")


def show_sample(df: pd.DataFrame, col: str = None, value: str = None, n: int = 10):
    """Filter and show rows."""
    if col and value:
        mask = df[col].astype(str).str.contains(value, case=False, na=False)
        filtered = df[mask]
        print(f"\n  Filtered: {col} contains '{value}' → {len(filtered):,} rows")
        if filtered.empty:
            print("  No matching rows.")
            return
        print(filtered.head(n).to_string(index=False))
    else:
        print(df.sample(min(n, len(df))).to_string(index=False))


def interactive_menu():
    """Main interactive loop."""
    print("\n" + "=" * 60)
    print("  Virtual Choreographies — Data Inspector")
    print("=" * 60)

    # Check which files exist
    available = {}
    for name, fname in FILES.items():
        path = os.path.join(DATA_DIR, fname)
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            available[name] = size_mb
            print(f"  ✓ {fname:<25} {size_mb:>8.1f} MB")
        else:
            print(f"  ✗ {fname:<25} not found")

    if not available:
        print("\n  No data files found. Run generate_synthetic_data.py first.")
        return

    while True:
        print(f"\n{'─' * 40}")
        print("  Available tables:")
        for i, name in enumerate(available, 1):
            print(f"    {i}. {name}")
        print(f"    q. Quit")
        print(f"{'─' * 40}")

        choice = input("  Select table (number or name): ").strip().lower()

        if choice in ("q", "quit", "exit"):
            print("  Bye!")
            break

        # Resolve choice
        table_name = None
        if choice.isdigit():
            idx = int(choice) - 1
            names = list(available.keys())
            if 0 <= idx < len(names):
                table_name = names[idx]
        elif choice in available:
            table_name = choice

        if not table_name:
            print("  Invalid choice. Try again.")
            continue

        print(f"\n  Loading {table_name}...")
        df = load_table(table_name)
        if df is None:
            continue

        show_overview(df, table_name)

        # Sub-menu for this table
        while True:
            print(f"\n  [{table_name}] What to show?")
            print(f"    h  — First 10 rows (head)")
            print(f"    t  — Last 10 rows (tail)")
            print(f"    s  — Numeric statistics")
            print(f"    c  — Categorical value counts")
            print(f"    r  — Random sample (10 rows)")
            print(f"    f  — Filter by column value")
            print(f"    n  — Show N rows from start")
            print(f"    b  — Back to table selection")

            sub = input("  > ").strip().lower()

            if sub == "b":
                break
            elif sub == "h":
                show_head(df, 10)
            elif sub == "t":
                pd.set_option("display.max_columns", None)
                pd.set_option("display.width", None)
                pd.set_option("display.max_colwidth", 40)
                print(f"\nLast 10 rows:")
                print(df.tail(10).to_string(index=False))
            elif sub == "s":
                show_stats(df)
            elif sub == "c":
                show_categoricals(df)
            elif sub == "r":
                pd.set_option("display.max_columns", None)
                pd.set_option("display.width", None)
                pd.set_option("display.max_colwidth", 40)
                print(f"\nRandom sample:")
                show_sample(df)
            elif sub == "f":
                print(f"  Columns: {', '.join(df.columns)}")
                col = input("  Column name: ").strip()
                if col not in df.columns:
                    print(f"  Column '{col}' not found.")
                    continue
                val = input("  Contains value: ").strip()
                pd.set_option("display.max_columns", None)
                pd.set_option("display.width", None)
                pd.set_option("display.max_colwidth", 40)
                show_sample(df, col, val, 15)
            elif sub == "n":
                try:
                    n = int(input("  How many rows? ").strip())
                    show_head(df, n)
                except ValueError:
                    print("  Enter a number.")
            else:
                print("  Unknown option.")


def print_all_summaries():
    """Quick summary of all tables (non-interactive)."""
    for name in FILES:
        df = load_table(name)
        if df is not None:
            show_overview(df, name)
            show_head(df, 5)
            print()


def main():
    parser = argparse.ArgumentParser(description="Inspect synthetic dataset")
    parser.add_argument("--file", type=str, default=None,
                        choices=list(FILES.keys()),
                        help="Jump to a specific table")
    parser.add_argument("--all", action="store_true",
                        help="Print summary of all tables and exit")
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="Directory with CSV files (default: ./data)")
    args = parser.parse_args()

    global DATA_DIR
    DATA_DIR = args.data_dir

    if args.all:
        print_all_summaries()
    elif args.file:
        df = load_table(args.file)
        if df is not None:
            show_overview(df, args.file)
            show_head(df, 10)
            show_stats(df)
            show_categoricals(df)
    else:
        interactive_menu()


if __name__ == "__main__":
    main()