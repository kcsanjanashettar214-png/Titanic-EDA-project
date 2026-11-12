"""Exploratory data analysis for the Titanic passenger dataset."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIRECTORY = PROJECT_ROOT / "data"
OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs"
DEFAULT_DATA_PATH = DATA_DIRECTORY / "titanic.csv"
DEFAULT_OUTPUT_DIR = OUTPUT_DIRECTORY
REQUIRED_COLUMNS = {"survived", "pclass", "sex", "age", "fare", "embarked"}
MAX_CSV_SIZE_BYTES = 10 * 1024 * 1024


def _resolve_project_path(path: Path, allowed_directory: Path, label: str) -> Path:
    """Resolve a path and ensure it stays inside its designated project directory."""
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate

    try:
        resolved_path = candidate.resolve(strict=label == "dataset")
        allowed_root = allowed_directory.resolve(strict=True)
    except (OSError, RuntimeError):
        raise ValueError(f"The {label} path is unavailable.") from None

    try:
        resolved_path.relative_to(allowed_root)
    except ValueError:
        raise ValueError(
            f"The {label} path must stay inside the project's "
            f"{allowed_directory.name} folder."
        ) from None

    return resolved_path


def load_dataset(data_path: Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load and validate a reasonably sized CSV from the project's data folder."""
    candidate = Path(data_path)
    if candidate.suffix.lower() != ".csv":
        raise ValueError(
            "Unsupported dataset format. Select a .csv file from the project's data folder."
        )

    resolved_path = _resolve_project_path(candidate, DATA_DIRECTORY, "dataset")
    if not resolved_path.is_file():
        raise ValueError("The selected dataset is not a regular file.")

    try:
        file_size = resolved_path.stat().st_size
    except OSError:
        raise ValueError("The selected dataset cannot be accessed.") from None

    if file_size > MAX_CSV_SIZE_BYTES:
        raise ValueError("The CSV is too large. The maximum supported size is 10 MB.")

    try:
        with resolved_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
            reader = csv.reader(csv_file, strict=True)
            header = next(reader, None)
            if not header:
                raise ValueError("The selected CSV is empty or has no header row.")

            normalized_header = [column.strip().lower() for column in header]
            if len(normalized_header) != len(set(normalized_header)):
                raise ValueError("The CSV contains duplicate column names.")

            for row in reader:
                if row and len(row) != len(header):
                    raise ValueError("The selected file is not a valid, readable CSV.")

        dataframe = pd.read_csv(
            resolved_path,
            encoding="utf-8-sig",
            on_bad_lines="error",
        )
    except (csv.Error, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError):
        raise ValueError("The selected file is not a valid, readable CSV.") from None
    except OSError:
        raise ValueError("The selected dataset cannot be accessed.") from None

    dataframe.columns = dataframe.columns.str.strip().str.lower()
    missing_columns = REQUIRED_COLUMNS.difference(dataframe.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"The CSV is missing required analysis columns: {missing_list}."
        )

    if dataframe.empty:
        raise ValueError("The CSV contains no passenger records.")

    try:
        for column in ("survived", "pclass", "age", "fare"):
            dataframe[column] = pd.to_numeric(dataframe[column], errors="raise")
    except (TypeError, ValueError):
        raise ValueError(
            "The CSV contains invalid values in a required numeric column."
        ) from None

    if dataframe["survived"].isna().any() or not dataframe["survived"].isin([0, 1]).all():
        raise ValueError("The 'survived' column must contain only 0 or 1 values.")
    if dataframe["pclass"].isna().any() or not dataframe["pclass"].isin([1, 2, 3]).all():
        raise ValueError("The 'pclass' column must contain only 1, 2, or 3.")
    if not dataframe["sex"].dropna().isin(["female", "male"]).all():
        raise ValueError("The 'sex' column contains unsupported categories.")

    return dataframe


def clean_dataset(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove duplicate rows and fill missing Age, Embarked, and Fare values."""
    cleaned = dataframe.copy()
    duplicate_count = int(cleaned.duplicated().sum())
    cleaned = cleaned.drop_duplicates().copy()

    for column in ("age", "fare"):
        median_value = cleaned[column].median()
        if pd.isna(median_value):
            raise ValueError(f"Cannot fill {column!r}: the column has no valid values.")
        cleaned[column] = cleaned[column].fillna(median_value)

    embarked_mode = cleaned["embarked"].mode(dropna=True)
    if embarked_mode.empty:
        raise ValueError("Cannot fill 'embarked': the column has no valid values.")
    cleaned["embarked"] = cleaned["embarked"].fillna(embarked_mode.iloc[0])

    return cleaned, duplicate_count


def create_group_summaries(dataframe: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Create the requested survival, age, passenger-count, and fare summaries."""
    return {
        "survival_by_gender": dataframe.groupby("sex", observed=True)["survived"]
        .mean()
        .rename("mean_survival_rate")
        .to_frame(),
        "survival_by_class": dataframe.groupby("pclass", observed=True)["survived"]
        .mean()
        .rename("mean_survival_rate")
        .to_frame(),
        "average_age_by_gender": dataframe.groupby("sex", observed=True)["age"]
        .mean()
        .rename("average_age")
        .to_frame(),
        "passenger_count_by_class": dataframe.groupby("pclass", observed=True)
        .size()
        .rename("passenger_count")
        .to_frame(),
        "average_fare_by_class": dataframe.groupby("pclass", observed=True)["fare"]
        .mean()
        .rename("average_fare")
        .to_frame(),
        "survival_by_embarkation": dataframe.groupby("embarked", observed=True)[
            "survived"
        ]
        .agg(["mean", "count"])
        .rename(columns={"mean": "mean_survival_rate", "count": "passenger_count"}),
    }


def create_key_findings(dataframe: pd.DataFrame) -> list[str]:
    """Calculate concise findings from the cleaned data, rather than hard-coding them."""
    survival_rate = dataframe["survived"].mean()
    gender_rates = dataframe.groupby("sex", observed=True)["survived"].mean()
    class_rates = dataframe.groupby("pclass", observed=True)["survived"].mean()
    port_rates = dataframe.groupby("embarked", observed=True)["survived"].mean()
    fare_by_class = dataframe.groupby("pclass", observed=True)["fare"].mean()
    median_age = float(np.nanmedian(dataframe["age"].to_numpy()))
    class_counts = dataframe.groupby("pclass", observed=True).size()

    findings = [
        f"Overall, {survival_rate:.1%} of passengers in this dataset survived.",
        (
            f"The survival rate was {gender_rates.get('female', float('nan')):.1%} "
            f"for female passengers and {gender_rates.get('male', float('nan')):.1%} "
            "for male passengers."
        ),
        (
            f"Survival rates by passenger class were "
            f"{class_rates.get(1, float('nan')):.1%} in first class, "
            f"{class_rates.get(2, float('nan')):.1%} in second class, and "
            f"{class_rates.get(3, float('nan')):.1%} in third class."
        ),
        f"The median passenger age after filling missing ages was {median_age:.1f} years.",
        (
            f"Class {int(class_counts.idxmax())} had the largest passenger count "
            f"({int(class_counts.max())} passengers)."
        ),
        (
            f"Average fare was highest in class {int(fare_by_class.idxmax())} "
            f"(${fare_by_class.max():.2f})."
        ),
        (
            f"Among the recorded embarkation ports, {port_rates.idxmax()} had the "
            f"highest observed survival rate ({port_rates.max():.1%})."
        ),
    ]
    return findings


def create_charts(dataframe: pd.DataFrame, charts_dir: Path) -> list[Path]:
    """Save six labeled charts and return their file paths."""
    charts_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    saved_paths: list[Path] = []

    def save_chart(filename: str) -> None:
        chart_path = charts_dir / filename
        plt.tight_layout()
        plt.savefig(chart_path, dpi=160, bbox_inches="tight")
        plt.close()
        saved_paths.append(chart_path)

    plt.figure(figsize=(7, 5))
    sns.countplot(data=dataframe, x="survived", hue="survived", palette="Set2")
    plt.title("Titanic Passenger Survival Counts")
    plt.xlabel("Survived (0 = No, 1 = Yes)")
    plt.ylabel("Number of passengers")
    plt.legend(title="Outcome", labels=["Did not survive", "Survived"])
    save_chart("survival_count.png")

    plt.figure(figsize=(8, 5))
    sns.histplot(data=dataframe, x="age", bins=30, kde=True, color="#4472C4")
    plt.title("Passenger Age Distribution")
    plt.xlabel("Age (years)")
    plt.ylabel("Number of passengers")
    save_chart("age_distribution.png")

    plt.figure(figsize=(7, 5))
    sns.barplot(
        data=dataframe,
        x="sex",
        y="survived",
        hue="sex",
        order=["female", "male"],
        hue_order=["female", "male"],
        palette="Set2",
        errorbar=None,
    )
    plt.title("Survival Rate by Gender")
    plt.xlabel("Gender")
    plt.ylabel("Mean survival rate")
    plt.ylim(0, 1)
    save_chart("gender_survival.png")

    plt.figure(figsize=(7, 5))
    sns.barplot(
        data=dataframe,
        x="pclass",
        y="survived",
        hue="pclass",
        order=[1, 2, 3],
        hue_order=[1, 2, 3],
        palette="Blues",
        errorbar=None,
    )
    plt.title("Survival Rate by Passenger Class")
    plt.xlabel("Passenger class (1 = first, 3 = third)")
    plt.ylabel("Mean survival rate")
    plt.ylim(0, 1)
    save_chart("class_survival.png")

    numeric_dataframe = dataframe.select_dtypes(include=np.number)
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        numeric_dataframe.corr(),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
    )
    plt.title("Correlation Heatmap of Numeric Features")
    save_chart("correlation_heatmap.png")

    plt.figure(figsize=(7, 5))
    sns.boxplot(data=dataframe, x="age", color="#70AD47")
    plt.title("Passenger Age Distribution and Outliers")
    plt.xlabel("Age (years)")
    save_chart("age_boxplot.png")

    return saved_paths


def run_analysis(data_path: Path, output_dir: Path) -> None:
    """Run the full analysis and write the cleaned data, charts, and summaries."""
    raw_dataframe = load_dataset(data_path)
    safe_output_dir = _resolve_project_path(
        output_dir, OUTPUT_DIRECTORY, "output"
    )
    print("=== Dataset exploration ===")
    print(f"Shape: {raw_dataframe.shape}")
    print(f"Columns: {raw_dataframe.columns.tolist()}")
    print("\nFirst five rows:")
    print(raw_dataframe.head().to_string(index=False))
    print("\nLast five rows:")
    print(raw_dataframe.tail().to_string(index=False))
    print("\nDataFrame information:")
    raw_dataframe.info()
    print("\nNumeric descriptive statistics:")
    print(raw_dataframe.describe().to_string())
    print(
        "\nNumerical columns:",
        raw_dataframe.select_dtypes(include=np.number).columns.tolist(),
    )
    print(
        "Categorical columns:",
        raw_dataframe.select_dtypes(include=["object", "category", "bool"]).columns.tolist(),
    )
    print("\nMissing values before cleaning:")
    print(raw_dataframe.isna().sum().to_string())

    cleaned_dataframe, duplicate_count = clean_dataset(raw_dataframe)
    charts_dir = safe_output_dir / "charts"
    cleaned_path = safe_output_dir / "titanic_cleaned.csv"
    safe_output_dir.mkdir(parents=True, exist_ok=True)
    cleaned_dataframe.to_csv(cleaned_path, index=False)

    print("=== Dataset overview ===")
    print(f"Duplicate rows removed: {duplicate_count}")
    print(f"Rows and columns after cleaning: {cleaned_dataframe.shape}")
    print("\nMissing values after cleaning:")
    print(cleaned_dataframe.isna().sum().to_string())

    print("\n=== Key statistical findings ===")
    for number, finding in enumerate(create_key_findings(cleaned_dataframe), start=1):
        print(f"{number}. {finding}")

    print("\n=== Group summaries ===")
    for name, summary in create_group_summaries(cleaned_dataframe).items():
        print(f"\n{name.replace('_', ' ').title()}:")
        print(summary.to_string(float_format=lambda value: f"{value:.3f}"))

    chart_paths = create_charts(cleaned_dataframe, charts_dir)
    print("\nCleaned dataset saved to outputs/titanic_cleaned.csv")
    print("Charts saved to outputs/charts/")
    print(f"Charts created: {len(chart_paths)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean and explore the Titanic passenger dataset."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="CSV file inside the project's data folder.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output folder inside the project's outputs folder.",
    )
    arguments = parser.parse_args()
    try:
        run_analysis(arguments.data, arguments.output_dir)
    except OSError:
        parser.exit(
            2,
            "Error: Analysis could not access an input or output file. "
            "Check the project folders and file permissions.\n",
        )
    except ValueError as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
