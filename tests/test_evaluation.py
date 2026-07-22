import pytest

from src.music_recommender.evaluation import hit_rate_at_k


def test_hit_rate_returns_one_when_relevant_item_is_recommended() -> None:
    recommendations = [10, 20, 30]
    relevant_items = {20}

    result = hit_rate_at_k(
        recommendations=recommendations,
        relevant_items=relevant_items,
        k=3,
    )

    assert result == 1.0


def test_hit_rate_returns_zero_when_no_relevant_item_is_recommended() -> None:
    recommendations = [10, 20, 30]
    relevant_items = {99}

    result = hit_rate_at_k(
        recommendations=recommendations,
        relevant_items=relevant_items,
        k=3,
    )

    assert result == 0.0


def test_hit_rate_only_checks_first_k_recommendations() -> None:
    recommendations = [10, 20, 30]
    relevant_items = {30}

    result = hit_rate_at_k(
        recommendations=recommendations,
        relevant_items=relevant_items,
        k=2,
    )

    assert result == 0.0


def test_hit_rate_rejects_non_positive_k() -> None:
    with pytest.raises(ValueError, match="k must be greater than zero"):
        hit_rate_at_k(
            recommendations=[10, 20],
            relevant_items={10},
            k=0,
        )