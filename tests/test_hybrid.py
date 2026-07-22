import pandas as pd
import pytest

from music_recommender.base import RecommenderModel
from music_recommender.models.hybrid import HybridRecommender


class FakeRecommender(RecommenderModel):
    """Small deterministic recommender used for hybrid unit tests."""

    def __init__(self, recommendations: list[int]) -> None:
        self.recommendations = recommendations
        self.is_fitted = False

    def fit(self, interactions: pd.DataFrame) -> None:
        self.is_fitted = True

    def recommend(
        self,
        user_id: int,
        k: int = 10,
    ) -> list[int]:
        if not self.is_fitted:
            raise RuntimeError("Fake model has not been fitted.")

        return self.recommendations[:k]


def create_sample_interactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": [1, 1],
            "item_id": [10, 20],
            "events": [1, 1],
        }
    )


def test_hybrid_can_fit_both_models() -> None:
    primary = FakeRecommender([30, 40])
    secondary = FakeRecommender([40, 50])

    model = HybridRecommender(primary, secondary)
    model.fit(create_sample_interactions())

    assert model.is_fitted is True
    assert primary.is_fitted is True
    assert secondary.is_fitted is True


def test_hybrid_combines_rankings() -> None:
    primary = FakeRecommender([30, 40, 50])
    secondary = FakeRecommender([40, 30, 60])

    model = HybridRecommender(
        primary_model=primary,
        secondary_model=secondary,
        primary_weight=0.5,
        secondary_weight=0.5,
    )
    model.fit(create_sample_interactions())

    recommendations = model.recommend(user_id=1, k=3)

    assert recommendations[0] in {30, 40}
    assert 30 in recommendations
    assert 40 in recommendations


def test_hybrid_recommend_before_fit_raises_error() -> None:
    model = HybridRecommender(
        FakeRecommender([30]),
        FakeRecommender([40]),
    )

    with pytest.raises(RuntimeError):
        model.recommend(user_id=1)


def test_hybrid_rejects_non_positive_k() -> None:
    model = HybridRecommender(
        FakeRecommender([30]),
        FakeRecommender([40]),
    )
    model.fit(create_sample_interactions())

    with pytest.raises(ValueError):
        model.recommend(user_id=1, k=0)


def test_hybrid_rejects_invalid_weights() -> None:
    with pytest.raises(ValueError):
        HybridRecommender(
            FakeRecommender([30]),
            FakeRecommender([40]),
            primary_weight=0.0,
            secondary_weight=0.0,
        )