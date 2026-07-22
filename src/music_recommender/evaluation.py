from collections.abc import Hashable, Sequence


def hit_rate_at_k(
    recommendations: Sequence[Hashable],
    relevant_items: set[Hashable],
    k: int = 10,
) -> float:
    """
    Calculate HitRate@K for one user.

    HitRate@K is equal to 1 if at least one relevant item appears
    in the first K recommendations; otherwise, it is equal to 0.

    Parameters
    ----------
    recommendations:
        Ranked list of recommended item identifiers.
    relevant_items:
        Ground-truth relevant items for the user.
    k:
        Number of top recommendations to evaluate.

    Returns
    -------
    float
        1.0 when a hit exists, otherwise 0.0.

    Raises
    ------
    ValueError
        If k is not positive.
    """
    if k <= 0:
        raise ValueError("k must be greater than zero.")

    top_k_recommendations = recommendations[:k]

    has_hit = any(
        item_id in relevant_items
        for item_id in top_k_recommendations
    )

    return float(has_hit)