# EDA Workflow

An AI-powered exploratory data analysis workflow that performs consistent, first-pass analysis of datasets using LangChain and LangGraph. The workflow runs a fixed set of analysis tools, uses an LLM to extract observations after each step, and synthesizes findings into a summary with actionable recommendations — then generates a multi-page PDF report with adaptive visualizations.

## How It Works

The workflow follows a sequential process:
1. **Detect Schema**: Auto-classifies every column into a semantic type (numeric, categorical, date, ID, text) — all downstream steps adapt to this schema
2. **Analyze**: Runs a fixed set of predefined analysis tools on the dataset
3. **Observe**: After each analysis tool, the LLM extracts concise observations from the results
4. **Synthesize**: Once all tools have run, the LLM summarizes findings and provides actionable recommendations
5. **Report**: Generates a multi-page PDF with distributions, trends, correlations, and the written synthesis

This approach combines deterministic pandas-based analysis with LLM-powered interpretation.

## Analysis Pipeline

The workflow executes these nodes in order:

| Node | What it does |
|---|---|
| `detect_schema` | Classifies columns: numeric, categorical, date, ID (>80% unique), or text (avg length >50). All downstream nodes use this schema to skip irrelevant columns. |
| `profile_dataset` | Shape, dtypes, `describe()` statistics for numeric columns, and top-10 value counts for categorical columns. |
| `validate_data_integrity` | Checks `total = quantity × price` relationships, flags negative values in quantity/count columns. Severity rated HIGH/MEDIUM. |
| `extract_observations` ×5 | LLM step: extracts 1–2 concise observations from each preceding analysis result. |
| `analyze_missingness` | Per-column null counts and percentages; flags columns with >20% missing. |
| `detect_temporal_anomalies` | If date columns exist: groups by date, computes daily sums, and flags values more than 1 std dev above the mean as peaks. |
| `compute_aggregates` | Group-by aggregations (sum/mean/count). Selects categorical columns with 3–50 unique values and numeric columns by highest variance. |
| `analyze_relationships` | Three relationship types: (1) numeric–numeric Pearson correlation (threshold 0.7), (2) categorical–numeric ANOVA F-statistic, (3) categorical–categorical cross-tabulation. ID-like columns are excluded from cross-tabs. |
| `synthesize_findings` | LLM step: produces a 2–3 sentence summary and 3–5 actionable recommendations from all observations. |
| `generate_report` | Builds a multi-page PDF: overview + summary, numeric histograms, box plots, temporal trends (if date columns exist), categorical bar charts, pie charts, and a correlation heatmap. |

## Setup

### Prerequisites

- **Python 3.10 or 3.11**
- **Poetry** (dependency manager)
- **OpenAI API Key**

### Installation Steps

1. **Install Poetry** (if not already installed):
   
   **Windows (PowerShell)**:
   ```powershell
   (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
   ```
   
   **macOS/Linux**:
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```
   
   After installation, restart your terminal. If `poetry` command is not found:
   - **Windows**: Add `%APPDATA%\Python\Scripts` to your system PATH
   - **macOS/Linux**: Add `export PATH="$HOME/.local/bin:$PATH"` to your `~/.bashrc` or `~/.zshrc`

2. **Install dependencies**:
   ```bash
   poetry install
   ```
   
   This will install all dependencies with the exact versions specified in `poetry.lock`, ensuring consistency across all environments.

3. **Set up your OpenAI API key**:
   
   **Windows**:
   ```powershell
   copy .env.example .env
   ```
   
   **macOS/Linux**:
   ```bash
   cp .env.example .env
   ```
   
   Then edit `.env` and add your OpenAI API key:
   ```
   OPENAI_API_KEY=sk-your-key-here
   ```

### Multiple Python Versions?

If you have multiple Python versions installed and want to use a specific one:

```bash
# Tell Poetry which Python to use
poetry env use python3.11  # or python3.10

# Then install dependencies
poetry install
```

Poetry will create a virtual environment with your chosen Python version.

## Usage

### Python API

```python
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from eda_workflow.eda_workflow import EDAWorkflow

load_dotenv()

# Initialize the workflow with an LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
workflow = EDAWorkflow(model=llm)

# Run analysis on a dataset (report_path is optional, defaults to "eda_report.pdf")
workflow.invoke_workflow("data/cafe_sales.csv", report_path="my_report.pdf")

# Retrieve results
summary = workflow.get_summary()              # str
recommendations = workflow.get_recommendations()  # list[str]
observations = workflow.get_observations()    # dict[str, list[str]]
results = workflow.get_results()              # dict
report_path = workflow.get_report_path()      # str — path to the generated PDF
```

### Running the Example

```bash
poetry run python example_usage.py
```

This runs a full analysis on the sample dataset, prints results for each step, and saves `eda_report.pdf` and a `graph.png` diagram of the workflow graph.

## Project Structure

```
eda-agent/
├── data/
│   └── cafe_sales.csv             # Sample dataset
├── eda_workflow/
│   ├── __init__.py
│   ├── eda_workflow.py             # Main workflow class, graph, and all node definitions
│   ├── eda_utils.py                # Schema detection and column classification utilities
│   └── prompts/                   # LLM prompt templates
│       ├── extract_observations_system.txt
│       ├── extract_observations_human.txt
│       ├── synthesize_findings_system.txt
│       └── synthesize_findings_human.txt
├── .env.example                   # Environment variable template
├── example_usage.py               # Example script
├── pyproject.toml                 # Dependencies configuration
├── poetry.lock                    # Locked dependency versions
└── README.md
```

**Important**: The `poetry.lock` file is committed to ensure all users get identical, tested dependency versions.

## Dependencies

| Package | Purpose |
|---|---|
| `langchain`, `langchain-openai` | LLM prompt templates and OpenAI integration |
| `langgraph` | Workflow graph orchestration |
| `pandas` | Data loading and all deterministic analysis |
| `scipy` | ANOVA F-statistic for categorical–numeric relationships |
| `matplotlib`, `seaborn` | PDF report generation and visualizations |
| `python-dotenv` | `.env` file loading |
