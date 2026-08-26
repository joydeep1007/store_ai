# Retail Operations AI

## Project Status

Initial scaffold

## Description

A Python-based retail operations analytics and AI reporting project.

## Project Structure

- `data/`: Holds raw input CSV data files (`stores.csv`, `transactions.csv`, `staffing_shifts.csv`, `returns.csv`). The raw assignment CSV files will be manually placed inside `data/`. Raw input data should not be modified directly.
- `src/`: Core Python source code and modules (e.g., data quality auditing, data cleaning, KPI metrics, AI reporting).
- `tests/`: Automated tests and test suites for validating data integrity and pipeline functions.
- `notebooks/`: Jupyter notebooks for exploratory data analysis (`exploration.ipynb`) and experimentation.
- `outputs/`: Output directory for generated reports, audit logs, cleaned datasets, and visualizations.

## Setup

Create a virtual environment:

```bash
python -m venv .venv
```

Windows activation:

```cmd
.venv\Scripts\activate
```

Installing dependencies:

```bash
pip install -r requirements.txt
```

Running the project:

```bash
python main.py
```
