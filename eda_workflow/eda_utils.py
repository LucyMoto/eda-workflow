"""
Utilities for EDA workflow - schema detection and analysis helpers.

Provides functions to automatically classify columns and adapt analyses
to any dataset schema.
"""

import logging
import pandas as pd

logger = logging.getLogger(__name__)


def detect_column_schema(df: pd.DataFrame) -> dict:
    """
    Automatically classify columns into semantic types.
    
    This function examines each column and determines if it's:
    - ID column: unique identifier (>80% unique values)
    - Numeric: numeric data for statistical analysis
    - Categorical: low-cardinality text for grouping (3-50 unique)
    - Date: temporal data (parseable as datetime)
    - Text: high-cardinality text (avg length >50 or <5 unique)
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataset to analyze
    
    Returns
    -------
    dict
        Schema classification with keys:
        - id_columns: list of identifier columns
        - numeric_columns: list of numeric columns
        - categorical_columns: list of categorical columns
        - date_columns: list of datetime columns
        - text_columns: list of text columns
    """
    schema = {
        'id_columns': [],
        'numeric_columns': [],
        'categorical_columns': [],
        'date_columns': [],
        'text_columns': [],
    }
    
    for col in df.columns:
        dtype = df[col].dtype
        unique_count = df[col].nunique()
        unique_pct = unique_count / len(df) if len(df) > 0 else 0
        
        # Helper: check if column is string-like (object or StringDtype)
        is_string_like = pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype)
        
        # ========== NUMERIC COLUMNS (FIRST - no ambiguity) ==========
        if pd.api.types.is_numeric_dtype(dtype):
            schema['numeric_columns'].append(col)
            logger.debug(f"Classified '{col}' as numeric column")
            continue
        
        # ========== DATE COLUMNS (check BEFORE ID/text) ==========
        # Only if string type AND contains date-like patterns
        if is_string_like:
            try:
                # Try to parse as datetime
                parsed = pd.to_datetime(df[col], format='mixed', errors='coerce')
                # Check if at least 80% of values successfully parsed
                success_rate = parsed.notna().sum() / len(df)
                if success_rate > 0.8:
                    schema['date_columns'].append(col)
                    logger.debug(f"Classified '{col}' as date column ({success_rate*100:.0f}% parsed)")
                    continue
            except (ValueError, TypeError):
                pass
        
        # ========== TEXT COLUMNS (check BEFORE ID - high length = text) ==========
        # String with high average length → text (before ID check to catch long strings)
        if is_string_like:
            avg_length = df[col].astype(str).str.len().mean()
            if avg_length > 50:  # Long strings = text
                schema['text_columns'].append(col)
                logger.debug(f"Classified '{col}' as text column (avg_length={avg_length:.1f})")
                continue
        
        # ========== ID COLUMNS ==========
        # >80% unique values, string type → likely an identifier
        # (but exclude very low cardinality - those are categorical)
        if unique_pct > 0.8 and is_string_like and unique_count > 5:
            schema['id_columns'].append(col)
            logger.debug(f"Classified '{col}' as ID column ({unique_pct*100:.1f}% unique)")
            continue
        
        # ========== TEXT vs CATEGORICAL ==========
        # String type with moderate cardinality
        if is_string_like:
            # Very few unique values → categorical
            if unique_count < 3:
                schema['categorical_columns'].append(col)
                logger.debug(f"Classified '{col}' as categorical column ({unique_count} unique)")
            else:
                # Moderate cardinality → categorical (good for grouping)
                schema['categorical_columns'].append(col)
                logger.debug(f"Classified '{col}' as categorical column ({unique_count} unique)")
            continue
        
        # ========== DEFAULT ==========
        # Unknown dtype → treat as categorical
        schema['categorical_columns'].append(col)
        logger.debug(f"Classified '{col}' as categorical (fallback)")
    
    logger.info(f"Schema detection complete:")
    logger.info(f"  ID columns: {schema['id_columns']}")
    logger.info(f"  Numeric columns: {schema['numeric_columns']}")
    logger.info(f"  Categorical columns: {schema['categorical_columns']}")
    logger.info(f"  Date columns: {schema['date_columns']}")
    logger.info(f"  Text columns: {schema['text_columns']}")
    
    return schema


def filter_high_cardinality_columns(df: pd.DataFrame, columns: list, threshold: float = 0.5) -> list:
    """
    Filter out ID-like columns (high cardinality) from a list.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataset
    columns : list
        Column names to filter
    threshold : float, default=0.5
        If unique_pct > threshold, exclude the column
    
    Returns
    -------
    list
        Filtered column names (excluding high-cardinality ID-like columns)
    """
    filtered = []
    for col in columns:
        unique_pct = df[col].nunique() / len(df) if len(df) > 0 else 0
        if unique_pct <= threshold:
            filtered.append(col)
        else:
            logger.debug(f"Excluded high-cardinality column '{col}' ({unique_pct*100:.1f}% unique)")
    
    return filtered


def get_column_properties(df: pd.DataFrame, col: str) -> dict:
    """
    Get detailed properties of a single column.
    
    Returns dict with: dtype, unique_count, unique_pct, missing_pct, cardinality_rank
    """
    unique_count = df[col].nunique()
    unique_pct = unique_count / len(df) if len(df) > 0 else 0
    missing_pct = df[col].isna().sum() / len(df) * 100 if len(df) > 0 else 0
    
    # Cardinality classification
    if unique_pct > 0.5:
        cardinality_rank = 'high'
    elif unique_pct > 0.05:
        cardinality_rank = 'medium'
    else:
        cardinality_rank = 'low'
    
    return {
        'dtype': str(df[col].dtype),
        'unique_count': int(unique_count),
        'unique_pct': round(unique_pct, 3),
        'missing_pct': round(missing_pct, 2),
        'cardinality_rank': cardinality_rank
    }
