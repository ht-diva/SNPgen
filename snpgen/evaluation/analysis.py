"""
Analysis utilities for filtering and processing evaluation results.
"""

from typing import List, Optional


def filter_model_list(
    model_list: List[str],
    include_prs: bool,
    include_prs_univariate: bool,
    prs_to_include: Optional[List[str]] = None,
    prs_univariate_to_include: Optional[List[str]] = None,
) -> List[str]:
    """Filter a list of model names based on inclusion criteria.

    Filters model names based on whether to include 'prs' and 'prs univariate'
    models, with optional specific model inclusion.

    Args:
        model_list: The list of model names to filter.
        include_prs: Whether to include 'prs' models (excluding 'prs univariate').
        include_prs_univariate: Whether to include 'prs univariate' models.
        prs_to_include: A list of specific 'prs' model names to include,
            regardless of the `include_prs` flag. If `None`, all 'prs' models
            (excluding 'prs univariate') will be included based on the
            `include_prs` flag.
        prs_univariate_to_include: A list of specific 'prs univariate' model
            names to include, regardless of the `include_prs_univariate` flag.
            If `None`, all 'prs univariate' models will be included based on
            the `include_prs_univariate` flag.

    Returns:
        The filtered list of model names.
    """
    filtered_list = []

    for model_name in model_list:
        if model_name.startswith('prs univariate') and include_prs_univariate:
            # Include 'prs univariate' models if the flag is True
            if prs_univariate_to_include is None or model_name in prs_univariate_to_include:
                filtered_list.append(model_name)
        elif model_name.startswith('prs') and include_prs and not model_name.startswith('prs univariate'):
            # Include 'prs' models (excluding 'prs univariate') if the flag is True
            if prs_to_include is None or model_name in prs_to_include:
                filtered_list.append(model_name)
        elif not model_name.startswith('prs'):
            # Include all non-'prs' models
            filtered_list.append(model_name)

    return filtered_list
