import logging
import os
from typing import Optional, TypedDict

import pandas as pd
from scipy.stats import f_oneway
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

from .eda_utils import detect_column_schema, filter_high_cardinality_columns

logger = logging.getLogger(__name__)
WORKFLOW_NAME = "eda_workflow"
LOG_PATH = os.path.join(os.getcwd(), "logs/")
PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")


def load_prompt(filename: str) -> str:
    """Load a prompt template from the prompts directory."""
    prompt_path = os.path.join(PROMPTS_DIR, filename)
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


class EDAWorkflow:
    """
    Exploratory Data Analysis workflow that performs consistent, first-pass analysis of datasets.
    
    Uses a fixed set of predefined analysis tools to produce structured, tabular outputs.
    Operates sequentially and deterministically through baseline EDA steps.
    
    Parameters
    ----------
    model : LLM, optional
        Language model for synthesizing findings.
    log : bool, default=False
        Whether to save analysis results to a file.
    log_path : str, optional
        Directory for log files.
    checkpointer : Checkpointer, optional
        LangGraph checkpointer for saving workflow state.
    
    Attributes
    ----------
    response : dict or None
        Stores the full response after invoke_workflow() is called.
    """
    
    def __init__(
        self,
        model=None,
        log=False,
        log_path=None,
        checkpointer: Optional[object] = None
    ):
        self.model = model
        self.log = log
        self.log_path = log_path
        self.checkpointer = checkpointer
        self.response = None
        self._compiled_graph = make_eda_baseline_workflow(
            model=model,
            log=log,
            log_path=log_path,
            checkpointer=checkpointer
        )
    
    def invoke_workflow(self, filepath: str, report_path: str = "eda_report.pdf", **kwargs):
        """
        Run EDA analysis on the provided dataset.

        Parameters
        ----------
        filepath : str
            Path to the dataset file.
        report_path : str, default="eda_report.pdf"
            Output path for the generated PDF report.
        **kwargs
            Additional arguments passed to the underlying graph invoke method.

        Returns
        -------
        None
            Results are stored in self.response and accessed via getter methods.
        """
        df = pd.read_csv(filepath)

        response = self._compiled_graph.invoke({
            "dataframe": df.to_dict(),
            "schema": {},
            "results": {},
            "observations": {},
            "current_step": "",
            "summary": "",
            "recommendations": [],
            "report_path": report_path,
        }, **kwargs)
        
        self.response = response
        return None
    
    def get_summary(self):
        """Retrieves the analysis summary."""
        if self.response:
            return self.response.get("summary")
    
    def get_recommendations(self):
        """Retrieves the recommendations."""
        if self.response:
            return self.response.get("recommendations")
    
    def get_results(self):
        """Retrieves the full analysis results."""
        if self.response:
            return self.response.get("results")
    
    def get_observations(self):
        """Retrieves all observations from analysis steps."""
        if self.response:
            return self.response.get("observations")

    def get_report_path(self):
        """Retrieves the path of the generated PDF report."""
        if self.response:
            return self.response.get("results", {}).get("generate_report", {}).get("path")


def make_eda_baseline_workflow(
    model=None,
    log=False,
    log_path=None,
    checkpointer: Optional[object] = None
):
    """
    Factory function that creates a compiled LangGraph workflow for baseline EDA.
    
    Performs automated first-pass analysis with fixed analysis steps.
    
    Parameters
    ----------
    model : LLM, optional
        Language model for synthesizing findings.
    log : bool, default=False
        Whether to save analysis results to a file.
    log_path : str, optional
        Directory for log files.
    checkpointer : Checkpointer, optional
        LangGraph checkpointer for saving workflow state.
    
    Returns
    -------
    CompiledStateGraph
        Compiled LangGraph workflow ready to process EDA requests.
    """
    if log:
        if log_path is None:
            log_path = LOG_PATH
        if not os.path.exists(log_path):
            os.makedirs(log_path)
    
    class EDAState(TypedDict):
        dataframe: dict
        schema: dict
        results: dict
        observations: dict[str, list[str]]
        current_step: str
        summary: str
        recommendations: list[str]
        report_path: str
    
    def detect_schema_node(state: EDAState):
        """
        Auto-detect column types and semantic meanings.
        This node runs FIRST and determines all downstream analyses.
        """
        logger.info("Detecting data schema")
        df = pd.DataFrame.from_dict(state.get("dataframe"))
        
        schema = detect_column_schema(df)
        
        logger.info("Schema detection complete")
        
        return {
            "schema": schema,
            "current_step": "detect_schema",
        }
    
    def profile_dataset_node(state: EDAState):
        """Generate dataset profile with basic statistics."""
        logger.info("Profiling dataset")
        df = pd.DataFrame.from_dict(state.get("dataframe"))
        schema = state.get("schema", {})
        results = state.get("results", {})
        
        # Use schema-detected numeric columns (skip IDs)
        numeric_cols = schema.get('numeric_columns', [])
        categorical_cols = schema.get('categorical_columns', [])
        
        profile = {
            "shape": {"rows": len(df), "columns": len(df.columns)},
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "numeric_summary": (
                df[numeric_cols].describe().to_dict() if numeric_cols else {}
            ),
            "categorical_summary": {
                col: df[col].value_counts().head(10).to_dict()
                for col in categorical_cols
            },
        }
        
        results["profile_dataset"] = profile
        
        return {
            "current_step": "profile_dataset",
            "results": results,
        }
    
    def validate_data_integrity_node(state: EDAState):
        """Detect data quality issues: invalid relationships, constraints, duplicates."""
        logger.info("Validating data integrity")
        df = pd.DataFrame.from_dict(state.get("dataframe"))
        schema = state.get("schema", {})
        results = state.get("results", {})

        integrity_issues = {}

        # ========== PATTERN 1: TOTAL = QUANTITY × PRICE ==========
        total_cols = [c for c in df.columns if any(x in c.lower() for x in ['total', 'amount'])]
        qty_cols = [c for c in df.columns if any(x in c.lower() for x in ['quantity', 'qty'])]
        price_cols = [c for c in df.columns if any(x in c.lower() for x in ['price', 'unit'])]

        if total_cols and qty_cols and price_cols:
            for total_col in total_cols:
                for qty_col in qty_cols:
                    for price_col in price_cols:
                        try:
                            expected = df[qty_col] * df[price_col]
                            actual = df[total_col]
                            tolerance = 0.01
                            denom = actual.abs() + 0.001
                            mismatch = (abs(expected - actual) / denom) > tolerance
                            mismatch_pct = mismatch.sum() / len(df) * 100

                            if mismatch_pct > 0:
                                integrity_issues[f"{total_col}_validation"] = {
                                    "expected_formula": f"{qty_col} × {price_col}",
                                    "mismatch_rows": int(mismatch.sum()),
                                    "mismatch_pct": round(mismatch_pct, 2),
                                    "severity": "HIGH" if mismatch_pct > 10 else "MEDIUM",
                                }
                                logger.warning(
                                    f"{mismatch_pct:.1f}% of rows: {total_col} ≠ {qty_col} × {price_col}"
                                )
                        except (TypeError, KeyError, ValueError) as e:
                            logger.warning(f"Failed integrity check {total_col}: {e}")

        # ========== PATTERN 2: NEGATIVE VALUES IN NON-NEGATIVE COLUMNS ==========
        for col in schema.get('numeric_columns', []):
            if any(x in col.lower() for x in ['quantity', 'count', 'units']):
                neg_count = int((df[col] < 0).sum())
                if neg_count > 0:
                    integrity_issues[f"{col}_negatives"] = {
                        "column": col,
                        "negative_count": neg_count,
                        "negative_pct": round(neg_count / len(df) * 100, 2),
                        "note": "Quantities should typically be positive",
                    }

        results["validate_data_integrity"] = integrity_issues

        return {
            "current_step": "validate_data_integrity",
            "results": results,
        }

    def analyze_missingness_node(state: EDAState):
        """Analyze missing values in the dataset."""
        logger.info("Analyzing missingness")
        df = pd.DataFrame.from_dict(state.get("dataframe"))
        results = state.get("results", {})
        
        missing_count = df.isnull().sum().to_dict()
        missing_pct = (
            (df.isnull().sum() / len(df) * 100).round(2).to_dict()
        )

        high_missing = {col: pct for col, pct in missing_pct.items() if pct > 20}

        _SENTINELS = {"", "n/a", "na", "null", "none", "#n/a", "-", "unknown", "?", "nan"}
        sentinel_counts = {}
        for col in df.columns:
            if pd.api.types.is_object_dtype(df[col]):
                n = int(df[col].astype(str).str.strip().str.lower().isin(_SENTINELS).sum())
                if n > 0:
                    sentinel_counts[col] = n

        missingness = {
            "total_rows": len(df),
            "missing_count": missing_count,
            "missing_percentage": missing_pct,
            "high_missing_columns": high_missing,
            "sentinel_value_counts": sentinel_counts,
            "complete_rows": int(df.dropna().shape[0]),
            "complete_rows_pct": (
                round(df.dropna().shape[0] / len(df) * 100, 2)
                if len(df) > 0 else 0
            ),
        }
        
        results["analyze_missingness"] = missingness
        
        return {
            "current_step": "analyze_missingness",
            "results": results,
        }


    def detect_temporal_anomalies_node(state: EDAState):
        """Detect peaks, trends, and anomalies in time series data."""
        logger.info("Detecting temporal anomalies")
        df = pd.DataFrame.from_dict(state.get("dataframe"))
        schema = state.get("schema", {})
        results = state.get("results", {})

        if not schema.get('date_columns'):
            logger.info("No date columns, skipping temporal analysis")
            results["detect_temporal_anomalies"] = {}
            return {"current_step": "detect_temporal_anomalies", "results": results}

        anomalies = {}
        date_col = schema['date_columns'][0]
        numeric_cols = schema.get('numeric_columns', [])

        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

        for num_col in numeric_cols:
            try:
                daily = df.groupby(date_col)[num_col].sum().sort_index()
                mean = daily.mean()
                std = daily.std()

                if std == 0 or pd.isna(std):
                    continue

                peaks = daily[daily > mean + std].sort_values(ascending=False)

                if len(peaks) > 0:
                    anomalies[num_col] = {
                        "peaks": {str(k): float(v) for k, v in peaks.head(5).items()},
                        "peak_dates": [str(d) for d in peaks.head(5).index],
                        "mean_daily": float(mean),
                        "std_daily": float(std),
                        "threshold": float(mean + std),
                    }
            except (TypeError, KeyError, ValueError) as e:
                logger.warning(f"Failed temporal analysis for {num_col}: {e}")

        results["detect_temporal_anomalies"] = anomalies

        return {
            "current_step": "detect_temporal_anomalies",
            "results": results,
        }

    def compute_aggregates_node(state: EDAState):
        """
        Compute group-by aggregations on meaningful columns.
        Adapts to whatever categorical and numeric columns exist in the schema.
        """
        logger.info("Computing aggregates")
        df = pd.DataFrame.from_dict(state.get("dataframe"))
        schema = state.get("schema", {})
        results = state.get("results", {})
        
        # Get schema-detected columns (skip ID columns)
        numeric_cols = schema.get('numeric_columns', [])
        categorical_cols = schema.get('categorical_columns', [])
        
        aggregates = {}
        
        # ========== SKIP IF NO NUMERIC COLUMNS ==========
        if not numeric_cols:
            logger.warning("No numeric columns in schema, skipping aggregates")
            return {
                "current_step": "compute_aggregates",
                "results": results | {"compute_aggregates": aggregates},
            }
        
        # ========== SKIP IF NO CATEGORICAL COLUMNS ==========
        if not categorical_cols:
            logger.warning("No categorical columns in schema, skipping group-by analysis")
            return {
                "current_step": "compute_aggregates",
                "results": results | {"compute_aggregates": aggregates},
            }
        
        # ========== SELECT CATEGORICAL COLUMNS BY CARDINALITY ==========
        cat_cardinality = {col: df[col].nunique() for col in categorical_cols}
        
        # Sweet spot: 3-50 unique values (not too granular, not too coarse)
        good_cat_cols = [
            col for col, card in cat_cardinality.items() 
            if 3 <= card <= 50
        ]
        
        # Fallback: if no columns in sweet spot, use columns closest to it
        if not good_cat_cols:
            # Fallback 1: Accept columns with at least 3 unique values (lower bar)
            minimum_cardinality_cols = [
                col for col, card in cat_cardinality.items() if card >= 3
            ]
            if minimum_cardinality_cols:
                good_cat_cols = minimum_cardinality_cols
                logger.info("Fallback: Using columns with cardinality >= 3")
            else:
                # Fallback 2: Last resort — all categorical columns (even binary)
                good_cat_cols = categorical_cols
                logger.info("Fallback: Using all categorical columns (may be low cardinality)")
        
        # Sort by cardinality (descending) and take top 3
        selected_cat_cols = sorted(
            good_cat_cols,
            key=lambda col: cat_cardinality[col],
            reverse=True
        )[:3]
        
        logger.info(f"Selected categorical columns for aggregation: {selected_cat_cols}")
        
        # ========== SELECT NUMERIC COLUMNS BY VARIANCE ==========
        VARIANCE_FLOOR = 0.01
        num_variance = df[numeric_cols].var()
        non_constant_cols = num_variance[num_variance > VARIANCE_FLOOR].dropna().index.tolist()
        
        if not non_constant_cols:
            logger.warning("All numeric columns are constant (no variance)")
            return {
                "current_step": "compute_aggregates",
                "results": results | {"compute_aggregates": aggregates},
            }
        
        # Sort by variance (highest first) and take top 3
        selected_num_cols = num_variance[non_constant_cols].nlargest(3).index.tolist()
        
        logger.info(f"Selected numeric columns for aggregation: {selected_num_cols}")
        
        # ========== COMPUTE AGGREGATES ==========
        for cat_col in selected_cat_cols:
            for num_col in selected_num_cols:
                try:
                    agg_result = df.groupby(cat_col)[num_col].agg(['sum', 'mean', 'count']).to_dict()
                    aggregates[f"{cat_col}_by_{num_col}"] = agg_result
                except (TypeError, KeyError, ValueError) as e:
                    logger.warning(f"Failed to aggregate {cat_col} by {num_col}: {e}")
                    continue
        
        # Compute totals for selected numeric columns
        totals = {col: float(df[col].sum()) for col in selected_num_cols}
        aggregates["totals"] = totals
        
        # Add metadata
        aggregates["_metadata"] = {
            "selected_categorical_cols": selected_cat_cols,
            "selected_numeric_cols": selected_num_cols,
            "categorical_cardinalities": {col: cat_cardinality[col] for col in selected_cat_cols},
            "numeric_variances": {col: float(num_variance[col]) for col in selected_num_cols},
            "total_categorical_cols": len(categorical_cols),
            "total_numeric_cols": len(numeric_cols),
        }
        
        agg_count = len([k for k in aggregates.keys() if k not in ('_metadata', 'totals')])
        logger.info(f"Computed {agg_count} group-by aggregations")
        
        return {
            "current_step": "compute_aggregates",
            "results": results | {"compute_aggregates": aggregates},
        }


    def analyze_relationships_node(state: EDAState):
        """
        Analyze relationships between variables.
        Uses schema-detected columns and filters ID-like columns.
        
        Three relationship types:
        1. Numeric-to-Numeric: Correlation (how much do columns move together?)
        2. Categorical-to-Numeric: Group differences (does category affect values?)
        3. Categorical-to-Categorical: Distribution (how are categories distributed?)
        """
        logger.info("Analyzing relationships")
        df = pd.DataFrame.from_dict(state.get("dataframe"))
        schema = state.get("schema", {})
        results = state.get("results", {})
        
        # Use schema-detected columns (skips ID columns)
        numeric_cols = schema.get('numeric_columns', [])
        categorical_cols = schema.get('categorical_columns', [])
        id_cols = schema.get('id_columns', [])
        
        relationships = {}
        
        # ========== 1. NUMERIC-to-NUMERIC CORRELATIONS ==========
        if len(numeric_cols) >= 2:
            try:
                correlations = df[numeric_cols].corr().to_dict()
                
                # Find strong correlations (> 0.7, excluding self-correlation)
                high_correlations = {}
                for col1 in numeric_cols:
                    for col2 in numeric_cols:
                        if col1 < col2:
                            corr_val = correlations[col1][col2]
                            if abs(corr_val) > 0.7:
                                high_correlations[f"{col1}_vs_{col2}"] = float(corr_val)
                
                relationships["numeric_correlations"] = {
                    "correlation_matrix": correlations,
                    "high_correlations": high_correlations,
                    "count": len(high_correlations)
                }
                logger.info(f"Found {len(high_correlations)} high correlations (> 0.7)")
            except Exception as e:
                logger.warning(f"Failed to compute numeric correlations: {e}")

        # ========== 2. CATEGORICAL-to-NUMERIC RELATIONSHIPS ==========
        if categorical_cols and numeric_cols:
            try:
                cat_numeric_rels = {}

                # Select MEANINGFUL categorical columns (3-50 cardinality sweet spot)
                cat_cardinality = {col: df[col].nunique() for col in categorical_cols}

                # Sweet spot: 3-50 unique values
                meaningful_cat_cols = [
                    col for col, card in cat_cardinality.items()
                    if 3 <= card <= 50
                ]

                # Fallback if none in sweet spot
                if not meaningful_cat_cols:
                    meaningful_cat_cols = sorted(
                        categorical_cols,
                        key=lambda col: cat_cardinality[col]
                    )

                # Take top 2 by cardinality
                cat_cols_to_analyze = sorted(
                    meaningful_cat_cols,
                    key=lambda col: cat_cardinality[col],
                    reverse=True
                )[:2]

                # Numeric columns: select high-variance ones
                num_cols_to_analyze = sorted(
                    numeric_cols,
                    key=lambda col: df[col].var(),
                    reverse=True
                )[:2]

                logger.info(f"Analyzing categorical cols: {cat_cols_to_analyze}")

                for cat_col in cat_cols_to_analyze:
                    for num_col in num_cols_to_analyze:
                        # Group means
                        group_means = df.groupby(cat_col)[num_col].mean().to_dict()

                        # F-statistic (effect size)
                        groups = [group[num_col].values for _, group in df.groupby(cat_col)]
                        # Filter out groups with 1 sample (needed for meaningful ANOVA)
                        valid_groups = [g for g in groups if len(g) > 1]
                        try:
                            if len(valid_groups) >= 2:
                                f_stat, p_value = f_oneway(*valid_groups)
                                effect_strength = "strong" if f_stat > 10 else "moderate" if f_stat > 3 else "weak"
                            else:
                                f_stat, p_value, effect_strength = None, None, "unknown"
                        except Exception:
                            f_stat, p_value, effect_strength = None, None, "unknown"

                        cat_numeric_rels[f"{cat_col}_by_{num_col}"] = {
                            "group_means": {k: float(v) for k, v in group_means.items()},
                            "f_statistic": float(f_stat) if f_stat else None,
                            "p_value": float(p_value) if p_value else None,
                            "effect_strength": effect_strength
                        }

                relationships["categorical_numeric_groups"] = cat_numeric_rels
                logger.info(f"Analyzed {len(cat_numeric_rels)} categorical-numeric relationships")
            except Exception as e:
                logger.warning(f"Failed to compute categorical-numeric relationships: {e}")
        
        # ========== 3. CATEGORICAL-to-CATEGORICAL RELATIONSHIPS ==========
        # CRITICAL: Filter out ID-like columns (high cardinality) before cross-tabulation
        meaningful_cat_for_crosstab = filter_high_cardinality_columns(df, categorical_cols, threshold=0.5)
        
        if len(meaningful_cat_for_crosstab) >= 2:
            try:
                cat_cat_rels = {}
                
                # Top 2 categorical columns by cardinality (excluding ID-like)
                cat_cols_to_analyze = sorted(
                    meaningful_cat_for_crosstab,
                    key=lambda col: df[col].nunique(),
                    reverse=True
                )[:2]
                
                logger.info(f"Categorical columns for crosstab (ID-like excluded): {cat_cols_to_analyze}")
                
                for i, col1 in enumerate(cat_cols_to_analyze):
                    for col2 in cat_cols_to_analyze[i+1:]:
                        # Cross-tabulation (counts)
                        crosstab = pd.crosstab(df[col1], df[col2])
                        crosstab_dict = crosstab.to_dict()
                        
                        # Percentage distribution (row-wise)
                        crosstab_pct = pd.crosstab(df[col1], df[col2], normalize='index') * 100
                        crosstab_pct_dict = {k: {k2: float(v2) for k2, v2 in v.items()} 
                                            for k, v in crosstab_pct.to_dict().items()}
                        
                        cat_cat_rels[f"{col1}_vs_{col2}"] = {
                            "counts": crosstab_dict,
                            "percentages_by_row": crosstab_pct_dict
                        }
                
                if cat_cat_rels:
                    relationships["categorical_distributions"] = cat_cat_rels
                    logger.info(f"Analyzed {len(cat_cat_rels)} categorical-categorical relationships")
                else:
                    logger.info("No meaningful categorical pairs for cross-tabulation")
            except Exception as e:
                logger.warning(f"Failed to compute categorical-categorical relationships: {e}")
        
        # ========== METADATA ==========
        relationships["_metadata"] = {
            "relationship_types_computed": [k for k in relationships.keys() if k != "_metadata"],
            "numeric_cols_analyzed": numeric_cols,
            "categorical_cols_analyzed": categorical_cols,
            "id_cols_excluded": id_cols,
            "correlation_threshold": 0.7,
            "f_statistic_thresholds": {"strong": 10, "moderate": 3},
            "sample_size": len(df),
        }
        
        rel_count = len([k for k in relationships.keys() if k != "_metadata"])
        logger.info(f"Computed {rel_count} relationship types")
        
        results["analyze_relationships"] = relationships
        
        return {
            "current_step": "analyze_relationships",
            "results": results,
        }    

    
    
    
    
    def extract_observations_node(state: EDAState):
        """Extract observations from the latest analysis results using LLM."""
        logger.info("Extracting observations")
        
        current_step = state.get("current_step", "")
        results = state.get("results", {})
        observations = state.get("observations", {})
        
        if model is None or not current_step or current_step not in results:
            return {"observations": observations}
        
        step_results = results.get(current_step, {})
        
        class ObservationOutput(BaseModel):
            observations: list[str] = Field(description="3-6 concise, actionable observations")
        
        observation_prompt = ChatPromptTemplate.from_messages([
            ("system", load_prompt("extract_observations_system.txt")),
            ("human", load_prompt("extract_observations_human.txt")),
        ])
        
        chain = observation_prompt | model.with_structured_output(ObservationOutput)
        MAX_CHARS = 20_000
        results_str = str(step_results)
        if len(results_str) > MAX_CHARS:
            results_str = results_str[:MAX_CHARS] + "\n...[truncated]"
        response = chain.invoke({
            "step_name": current_step.replace("_", " ").title(),
            "results": results_str
        })
        
        observations[current_step] = response.observations
        
        return {
            "observations": observations,
        }
    
    def synthesize_findings_node(state: EDAState):
        """Synthesize accumulated findings into summary and recommendations."""
        logger.info("Synthesizing findings")
        
        observations = state.get("observations", {})
        
        if model is None:
            return {
                "summary": "No LLM provided for synthesis",
                "recommendations": [],
            }
        
        class SynthesisOutput(BaseModel):
            summary: str = Field(description="A concise 2-3 sentence summary of key findings")
            recommendations: list[str] = Field(description="3-5 actionable recommendations")
        
        all_observations = []
        for step_name, step_obs in observations.items():
            all_observations.append(f"\n{step_name.replace('_', ' ').title()}:")
            for obs in step_obs:
                all_observations.append(f"  - {obs}")
        
        observations_text = "\n".join(all_observations)
        
        synthesis_prompt = ChatPromptTemplate.from_messages([
            ("system", load_prompt("synthesize_findings_system.txt")),
            ("human", load_prompt("synthesize_findings_human.txt")),
        ])
        
        chain = synthesis_prompt | model.with_structured_output(SynthesisOutput)
        response = chain.invoke({"observations": observations_text})
        
        return {
            "summary": response.summary,
            "recommendations": response.recommendations,
        }
    
    def generate_report_node(state: EDAState):
        """Generate a multi-page PDF report with adaptive visualizations."""
        import textwrap
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        import seaborn as sns

        logger.info("Generating PDF report")
        df = pd.DataFrame.from_dict(state.get("dataframe"))
        schema = state.get("schema", {})
        results = state.get("results", {})
        summary = state.get("summary", "")
        recommendations = state.get("recommendations", [])
        report_path = state.get("report_path") or "eda_report.pdf"

        numeric_cols = schema.get('numeric_columns', [])[:12]
        categorical_cols = schema.get('categorical_columns', [])[:8]
        date_cols = schema.get('date_columns', [])

        profile = results.get("profile_dataset", {})
        shape = profile.get("shape", {"rows": len(df), "columns": len(df.columns)})

        sns.set_theme(style="whitegrid", palette="muted")

        with PdfPages(report_path) as pdf:

            # ===== PAGE 1: OVERVIEW =====
            fig = plt.figure(figsize=(11, 8.5))
            ax = fig.add_axes([0, 0, 1, 1])
            ax.axis('off')
            fig.patch.set_facecolor('#F0F4F8')

            fig.text(0.5, 0.90, "Exploratory Data Analysis Report",
                     fontsize=24, fontweight='bold', ha='center', va='top', color='#1E3A5F')

            info = (
                f"{shape['rows']:,} rows  ·  {shape['columns']} columns  ·  "
                f"{len(numeric_cols)} numeric  ·  {len(categorical_cols)} categorical"
                + (f"  ·  {len(date_cols)} date" if date_cols else "")
            )
            fig.text(0.5, 0.80, info, fontsize=12, ha='center', va='top', color='#555555')

            y = 0.70
            if summary:
                fig.text(0.08, y, "Summary", fontsize=14, fontweight='bold', va='top', color='#1E3A5F')
                y -= 0.06
                for line in textwrap.wrap(summary, width=105):
                    fig.text(0.08, y, line, fontsize=10, va='top', color='#333333')
                    y -= 0.045

            if recommendations:
                y -= 0.03
                fig.text(0.08, y, "Recommendations", fontsize=14, fontweight='bold', va='top', color='#1E3A5F')
                y -= 0.06
                for rec in recommendations[:6]:
                    for j, line in enumerate(textwrap.wrap(rec, width=100)):
                        prefix = "·  " if j == 0 else "   "
                        fig.text(0.10, y, prefix + line, fontsize=10, va='top', color='#333333')
                        y -= 0.04

            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

            # ===== PAGE 2: NUMERIC DISTRIBUTIONS (HISTOGRAMS) =====
            if numeric_cols:
                n = len(numeric_cols)
                ncols = min(3, n)
                nrows = (n + ncols - 1) // ncols
                fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows), squeeze=False)
                fig.suptitle("Numeric Distributions", fontsize=16, fontweight='bold')
                axes_flat = axes.flatten()

                for i, col in enumerate(numeric_cols):
                    data = df[col].dropna()
                    ax = axes_flat[i]
                    ax.hist(data, bins=30, color='steelblue', edgecolor='white', alpha=0.85)
                    ax.axvline(data.mean(), color='#E74C3C', linestyle='--', linewidth=1.2,
                               label=f'Mean {data.mean():.2f}')
                    ax.axvline(data.median(), color='#F39C12', linestyle=':', linewidth=1.2,
                               label=f'Median {data.median():.2f}')
                    ax.set_title(col, fontsize=11)
                    ax.set_ylabel('Count')
                    ax.legend(fontsize=8)

                for j in range(i + 1, len(axes_flat)):
                    axes_flat[j].set_visible(False)

                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)

            # ===== PAGE 3: BOX PLOTS =====
            if numeric_cols:
                group_col = next(
                    (c for c in categorical_cols if 2 <= df[c].nunique() <= 8), None
                )
                n = len(numeric_cols)
                ncols = min(3, n)
                nrows = (n + ncols - 1) // ncols
                title = f"Box Plots — grouped by '{group_col}'" if group_col else "Box Plots"
                fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows), squeeze=False)
                fig.suptitle(title, fontsize=16, fontweight='bold')
                axes_flat = axes.flatten()

                for i, col in enumerate(numeric_cols):
                    ax = axes_flat[i]
                    if group_col:
                        data = df[[col, group_col]].dropna()
                        group_keys = sorted(data[group_col].unique())
                        groups = [data.loc[data[group_col] == k, col].values for k in group_keys]
                        bp = ax.boxplot(groups, labels=[str(k) for k in group_keys], patch_artist=True)
                        for patch in bp['boxes']:
                            patch.set_facecolor('steelblue')
                            patch.set_alpha(0.7)
                        ax.tick_params(axis='x', rotation=30)
                    else:
                        bp = ax.boxplot(df[col].dropna().values, patch_artist=True)
                        bp['boxes'][0].set_facecolor('steelblue')
                        bp['boxes'][0].set_alpha(0.7)
                    ax.set_title(col, fontsize=11)
                    ax.set_ylabel('Value')

                for j in range(i + 1, len(axes_flat)):
                    axes_flat[j].set_visible(False)

                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)

            # ===== PAGE 4: TEMPORAL TRENDS =====
            if date_cols and numeric_cols:
                date_col = date_cols[0]
                df_t = df.copy()
                df_t[date_col] = pd.to_datetime(df_t[date_col], errors='coerce')
                df_t = df_t.dropna(subset=[date_col])

                n = len(numeric_cols)
                ncols = min(2, n)
                nrows = (n + ncols - 1) // ncols
                fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows), squeeze=False)
                fig.suptitle(f"Temporal Trends  (by {date_col})", fontsize=16, fontweight='bold')
                axes_flat = axes.flatten()

                for i, col in enumerate(numeric_cols):
                    daily = df_t.groupby(date_col)[col].sum().sort_index()
                    ax = axes_flat[i]
                    ax.plot(daily.index, daily.values, color='steelblue', linewidth=1.5)
                    ax.fill_between(daily.index, daily.values, alpha=0.15, color='steelblue')
                    ax.set_title(col, fontsize=11)
                    ax.set_ylabel('Sum')
                    ax.tick_params(axis='x', rotation=30)

                for j in range(i + 1, len(axes_flat)):
                    axes_flat[j].set_visible(False)

                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)

            # ===== PAGE 5: CATEGORICAL BAR CHARTS =====
            if categorical_cols:
                n = len(categorical_cols)
                ncols = min(3, n)
                nrows = (n + ncols - 1) // ncols
                fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4.5 * nrows), squeeze=False)
                fig.suptitle("Categorical Distributions", fontsize=16, fontweight='bold')
                axes_flat = axes.flatten()

                for i, col in enumerate(categorical_cols):
                    counts = df[col].value_counts().head(10)
                    ax = axes_flat[i]
                    ax.barh(counts.index.astype(str), counts.values, color='steelblue', alpha=0.85)
                    ax.invert_yaxis()
                    ax.set_title(col, fontsize=11)
                    ax.set_xlabel('Count')

                for j in range(i + 1, len(axes_flat)):
                    axes_flat[j].set_visible(False)

                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)

            # ===== PAGE 6: PIE CHARTS (low-cardinality only) =====
            pie_cols = [c for c in categorical_cols if df[c].nunique() <= 10]
            if pie_cols:
                n = len(pie_cols)
                ncols = min(3, n)
                nrows = (n + ncols - 1) // ncols
                fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5 * nrows), squeeze=False)
                fig.suptitle("Category Proportions", fontsize=16, fontweight='bold')
                axes_flat = axes.flatten()

                for i, col in enumerate(pie_cols):
                    counts = df[col].value_counts().head(10)
                    ax = axes_flat[i]
                    ax.pie(counts.values, labels=counts.index.astype(str),
                           autopct='%1.1f%%', startangle=90,
                           colors=sns.color_palette("muted", len(counts)))
                    ax.set_title(col, fontsize=11)

                for j in range(i + 1, len(axes_flat)):
                    axes_flat[j].set_visible(False)

                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)

            # ===== PAGE 7: CORRELATION HEATMAP =====
            if len(numeric_cols) >= 2:
                corr = df[numeric_cols].corr()
                size = max(8, len(numeric_cols))
                fig, ax = plt.subplots(figsize=(size, size * 0.8))
                fig.suptitle("Correlation Heatmap", fontsize=16, fontweight='bold')
                sns.heatmap(
                    corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                    vmin=-1, vmax=1, ax=ax, square=True, linewidths=0.5,
                    annot_kws={'size': max(8, 12 - len(numeric_cols) // 2)},
                )
                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)

        logger.info(f"PDF report saved to {report_path}")
        results["generate_report"] = {"path": report_path}

        return {
            "current_step": "generate_report",
            "results": results,
        }

    workflow = StateGraph(EDAState)

    workflow.add_node("detect_schema", detect_schema_node)
    workflow.add_node("profile_dataset", profile_dataset_node)
    workflow.add_node("extract_observations_1", extract_observations_node)
    workflow.add_node("validate_data_integrity", validate_data_integrity_node)
    workflow.add_node("extract_observations_2", extract_observations_node)
    workflow.add_node("analyze_missingness", analyze_missingness_node)
    workflow.add_node("extract_observations_3", extract_observations_node)
    workflow.add_node("detect_temporal_anomalies", detect_temporal_anomalies_node)
    workflow.add_node("extract_observations_4", extract_observations_node)
    workflow.add_node("compute_aggregates", compute_aggregates_node)
    workflow.add_node("extract_observations_5", extract_observations_node)
    workflow.add_node("analyze_relationships", analyze_relationships_node)
    workflow.add_node("extract_observations_6", extract_observations_node)
    workflow.add_node("synthesize_findings", synthesize_findings_node)
    workflow.add_node("generate_report", generate_report_node)

    workflow.set_entry_point("detect_schema")

    workflow.add_edge("detect_schema", "profile_dataset")
    workflow.add_edge("profile_dataset", "extract_observations_1")
    workflow.add_edge("extract_observations_1", "validate_data_integrity")
    workflow.add_edge("validate_data_integrity", "extract_observations_2")
    workflow.add_edge("extract_observations_2", "analyze_missingness")
    workflow.add_edge("analyze_missingness", "extract_observations_3")
    workflow.add_edge("extract_observations_3", "detect_temporal_anomalies")
    workflow.add_edge("detect_temporal_anomalies", "extract_observations_4")
    workflow.add_edge("extract_observations_4", "compute_aggregates")
    workflow.add_edge("compute_aggregates", "extract_observations_5")
    workflow.add_edge("extract_observations_5", "analyze_relationships")
    workflow.add_edge("analyze_relationships", "extract_observations_6")
    workflow.add_edge("extract_observations_6", "synthesize_findings")
    workflow.add_edge("synthesize_findings", "generate_report")
    workflow.add_edge("generate_report", END)
    
    app = workflow.compile(checkpointer=checkpointer, name=WORKFLOW_NAME)
    
    return app
