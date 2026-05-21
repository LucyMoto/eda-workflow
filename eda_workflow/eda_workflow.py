import logging
import os
from typing import Optional, TypedDict

import pandas as pd
from scipy.stats import f_oneway
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

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
    
    def invoke_workflow(self, filepath: str, **kwargs):
        """
        Run EDA analysis on the provided dataset.
        
        Parameters
        ----------
        filepath : str
            Path to the dataset file.
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
            "results": {},
            "observations": {},
            "current_step": "",
            "summary": "",
            "recommendations": [],
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
        results: dict
        observations: dict[str, list[str]]
        current_step: str
        summary: str
        recommendations: list[str]
    
    def profile_dataset_node(state: EDAState):
        """Generate dataset profile with basic statistics."""
        logger.info("Profiling dataset")
        df = pd.DataFrame.from_dict(state.get("dataframe"))
        results = state.get("results", {})
        
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        
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
        
        missingness = {
            "total_rows": len(df),
            "missing_count": missing_count,
            "missing_percentage": missing_pct,
            "high_missing_columns": high_missing,
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


    def compute_aggregates_node(state: EDAState):
        """
        Compute aggregates on high-cardinality categorical and high-variance numeric columns.
        
        Strategy:
        - Categorical: Select 3-50 unique values (sweet spot for grouping)
        - Numeric: Select columns with highest variance (most signal)
        - Handle edge cases: no columns, all constant, etc.
        """
        logger.info("Computing aggregates")
        df = pd.DataFrame.from_dict(state.get("dataframe"))
        results = state.get("results", {})
        
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        
        aggregates = {}
        
        # ========== SELECT CATEGORICAL COLUMNS ==========
        if not categorical_cols:
            logger.warning("No categorical columns found")
            return {
                "current_step": "compute_aggregates",
                "results": results | {"compute_aggregates": aggregates},
            }
        
        # Cardinality analysis
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
        
        # ========== SELECT NUMERIC COLUMNS ==========
        if not numeric_cols:
            logger.warning("No numeric columns found")
            return {
                "current_step": "compute_aggregates",
                "results": results | {"compute_aggregates": aggregates},
            }
        
        # Variance analysis (exclude constant columns: variance = 0)
        num_variance = df[numeric_cols].var()
        # Only select columns with meaningful variance (>0.01 to avoid near-constant columns)
        VARIANCE_FLOOR = 0.01
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
                    # Safety net: shouldn't occur with proper column selection,
                    # but catches unexpected type mismatches or groupby failures
                    logger.warning(f"Failed to aggregate {cat_col} by {num_col}: {e}")
                    continue
        
        # Compute totals for selected numeric columns
        # Convert to float for JSON serialization (necessary for state persistence)
        totals = {col: float(df[col].sum()) for col in selected_num_cols}
        aggregates["totals"] = totals
        
        # Add metadata about what was selected (useful for debugging)
        aggregates["_metadata"] = {
            "selected_categorical_cols": selected_cat_cols,
            "selected_numeric_cols": selected_num_cols,
            "categorical_cardinalities": {col: cat_cardinality[col] for col in selected_cat_cols},
            "numeric_variances": {col: float(num_variance[col]) for col in selected_num_cols},
            "total_categorical_cols": len(categorical_cols),
            "total_numeric_cols": len(numeric_cols),
            "selection_reason": "High-cardinality categorical (3-50 unique), high-variance numeric",
        }
        
        # Count aggregations (exclude metadata and totals)
        agg_count = len([k for k in aggregates.keys() if k not in ('_metadata', 'totals')])
        logger.info(f"Computed {agg_count} group-by aggregations")
        
        return {
            "current_step": "compute_aggregates",
            "results": results | {"compute_aggregates": aggregates},
        }


    def analyze_relationships_node(state: EDAState):
        """
        Analyze relationships between variables.
        
        Three relationship types:
        1. Numeric-to-Numeric: Correlation (how much do columns move together?)
        2. Categorical-to-Numeric: Group differences (does category affect values?)
        3. Categorical-to-Categorical: Distribution (how are categories distributed?)
        """
        logger.info("Analyzing relationships")
        df = pd.DataFrame.from_dict(state.get("dataframe"))
        results = state.get("results", {})
        
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        
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
                    # NOT highest cardinality (which are unique identifiers)
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
                        )  # Sort ascending (lowest cardinality = most meaningful)
                    
                    # Take top 2 by cardinality (within sweet spot)
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
                            # Note: Small groups may inflate F-statistic; consider adding min_group_size=2+ check
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
        if len(categorical_cols) >= 2:
            try:
                cat_cat_rels = {}
                
                # Top 2 categorical columns by cardinality
                cat_cols_to_analyze = sorted(
                    categorical_cols,
                    key=lambda col: df[col].nunique(),
                    reverse=True
                )[:2]
                
                for i, col1 in enumerate(cat_cols_to_analyze):
                    for col2 in cat_cols_to_analyze[i+1:]:  # Avoid duplicates
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
                
                relationships["categorical_distributions"] = cat_cat_rels
                logger.info(f"Analyzed {len(cat_cat_rels)} categorical-categorical relationships")
            except Exception as e:
                logger.warning(f"Failed to compute categorical-categorical relationships: {e}")
        
        # ========== METADATA ==========
        relationships["_metadata"] = {
            "relationship_types_computed": [k for k in relationships.keys() if k != "_metadata"],
            "numeric_cols_analyzed": numeric_cols,
            "categorical_cols_analyzed": categorical_cols,
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
            observations: list[str] = Field(description="1-2 concise, actionable observations")
        
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
    
    workflow = StateGraph(EDAState)
    
    workflow.add_node("profile_dataset", profile_dataset_node)
    workflow.add_node("extract_observations_1", extract_observations_node)
    workflow.add_node("analyze_missingness", analyze_missingness_node)
    workflow.add_node("extract_observations_2", extract_observations_node)
    workflow.add_node("compute_aggregates", compute_aggregates_node)
    workflow.add_node("extract_observations_3", extract_observations_node)
    workflow.add_node("analyze_relationships", analyze_relationships_node)
    workflow.add_node("extract_observations_4", extract_observations_node)
    workflow.add_node("synthesize_findings", synthesize_findings_node)
    
    workflow.set_entry_point("profile_dataset")
    
    workflow.add_edge("profile_dataset", "extract_observations_1")
    workflow.add_edge("extract_observations_1", "analyze_missingness")
    workflow.add_edge("analyze_missingness", "extract_observations_2")
    workflow.add_edge("extract_observations_2", "compute_aggregates")
    workflow.add_edge("compute_aggregates", "extract_observations_3")
    workflow.add_edge("extract_observations_3", "analyze_relationships")
    workflow.add_edge("analyze_relationships", "extract_observations_4")
    workflow.add_edge("extract_observations_4", "synthesize_findings")
    workflow.add_edge("synthesize_findings", END)
    
    app = workflow.compile(checkpointer=checkpointer, name=WORKFLOW_NAME)
    
    return app
