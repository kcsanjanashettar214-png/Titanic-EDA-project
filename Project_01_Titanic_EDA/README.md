# Exploratory Data Analysis (EDA) on Titanic Dataset

A beginner-friendly Python data science project that explores passenger records from the Titanic dataset. It demonstrates how to inspect a CSV file, handle selected missing values and duplicate rows, summarize survival patterns, and save clear visualizations.

## Project goals

- Inspect the dimensions, columns, data types, and summary statistics of a real dataset.
- Clean missing `Age`, `Embarked`, and `Fare` values and remove exact duplicate rows.
- Compare survival rates by gender, passenger class, and embarkation port.
- Create six visualizations and calculate findings from the loaded data.
- Practice Pandas, NumPy, Matplotlib, Seaborn, and Jupyter Notebook workflows.

## Project structure

```text
Project_01_Titanic_EDA/
├── index.html                   # local visual project overview
├── data/
│   └── titanic.csv
├── notebooks/
│   └── titanic_eda.ipynb
├── src/
│   └── eda.py
├── outputs/
│   ├── charts/
│   └── titanic_cleaned.csv       # generated when the analysis runs
├── requirements.txt
├── SECURITY.md                   # input validation and data protection notes
└── README.md
```

## View the project website

Open `index.html` in a browser to see the project's responsive overview dashboard, findings, and charts. The dashboard shows aggregate statistics and visualizations; it does not display passenger-level records or link to a cleaned-CSV download. The website is a self-contained local report: it needs no web server, JavaScript package installation, or internet connection. Keep the `outputs/charts/` folder beside `index.html` so the chart images load correctly.

**Privacy limitation:** this is a static local website, not an access-control system. The original source CSV is publicly available, and the analysis script creates `outputs/titanic_cleaned.csv` locally. Anyone who receives the full project folder can inspect those files, so do not treat hiding the dashboard's download link as encryption or as a way to control access. Share only the dashboard and chart files when row-level data should not be distributed.

## Dataset

The included CSV is the passenger-identified Titanic training dataset from the public [Data Science Dojo datasets repository](https://github.com/datasciencedojo/datasets/blob/master/titanic.csv). It contains the `PassengerId` field, so passengers with otherwise identical recorded details are not mistakenly treated as duplicate records. The loader converts column names to lowercase for consistent Python access. The analysis uses the existing `Survived` field (`0` = did not survive, `1` = survived); it does not build a prediction model.

## Setup in VS Code (Windows)

1. Open the `Project_01_Titanic_EDA` folder in VS Code.
2. Open **Terminal → New Terminal**.
3. Create and activate a virtual environment:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, use the VS Code Python interpreter picker to select `.venv\Scripts\python.exe`, then run commands with `.\.venv\Scripts\python.exe`.

4. Install the project dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

5. In VS Code, install the **Python** and **Jupyter** extensions if they are not already installed. Select the `.venv` interpreter/kernel.

## Run the analysis

From the project root, run:

```powershell
python .\src\eda.py
```

The script checks that the CSV exists and that the required analysis columns are available. If `data\titanic.csv` is missing, it raises a clear error with instructions rather than silently continuing.

To select another input file or output folder, use locations inside the project's `data/` and `outputs/` folders:

```powershell
python .\src\eda.py --data .\data\titanic.csv --output-dir .\outputs
```

The loader accepts CSV files up to 10 MB from `data/`, checks required columns and key values, and rejects paths outside the project input/output folders. Invalid files produce a brief error without displaying their contents or full system paths. See [SECURITY.md](./SECURITY.md) for the security practices and limitations.

## Run the notebook

Open `notebooks/titanic_eda.ipynb` in VS Code and choose **Run All**. The notebook walks through the same analysis in small, explained sections. It imports the shared functions from `src/eda.py`, so the cleaning and chart logic stays consistent with the command-line script.

## Cleaning approach

- Report exact duplicate rows and remove them before analysis.
- Keep passenger identifiers in the source so distinct passengers with matching details are not removed as duplicates.
- Fill missing `Age` and `Fare` with their respective medians.
- Fill missing `Embarked` with its most frequent non-missing value (mode).
- Save the resulting table to `outputs/titanic_cleaned.csv`.
- Display missing values that remain in other source columns; this project intentionally limits imputation to the three requested fields instead of inventing values for unrelated columns.
- In the included CSV, `Age` has 177 missing values, `Embarked` has 2, and `Fare` has none. The `Fare` median-imputation step is implemented and will fill null fares in alternate input files, but makes no changes to this particular dataset.

## Analysis and visualizations

The project reports overall survival, survival rates by gender, class, and embarkation port, age distribution, passenger counts by class, and average fare. Pandas `groupby()` summaries include:

- Mean survival rate by gender and passenger class.
- Average age by gender.
- Passenger count by passenger class.
- Average fare by passenger class.
- Survival rate and passenger count by embarkation port.

Six charts are saved in `outputs/charts/`:

1. `survival_count.png` — passenger counts by survival outcome.
2. `age_distribution.png` — age histogram with a density curve.
3. `gender_survival.png` — mean survival rate by gender.
4. `class_survival.png` — mean survival rate by passenger class.
5. `correlation_heatmap.png` — correlations among numeric columns.
6. `age_boxplot.png` — age spread and potential outliers.

The console and notebook generate at least five findings from the loaded, cleaned dataset each time they run. Findings are calculated from the data rather than hard-coded.

### Findings from the included CSV

The included 891-row dataset produces these findings when run with the cleaning steps above:

- Overall survival rate: **38.4%**.
- Survival rate by gender: **74.2% for female passengers** and **18.9% for male passengers**.
- Survival by class: **63.0%** in first class, **47.3%** in second class, and **24.2%** in third class.
- Median age after filling missing ages: **28.0 years**.
- Third class had the most passengers: **491**.
- Average fare was highest in first class: **$84.15**.
- Embarkation port **C** had the highest observed survival rate among the recorded ports: **55.4%**.

## Notes

- This is an exploratory analysis of an historical dataset; observed associations do not establish causes.
- `pclass` is represented numerically but is an ordered passenger-class category.
- The project is local analysis code; it does not claim to be deployed.
