import pandas as pd
import pytest

from music_recommender.models.content_based import (
    ContentBasedRecommender,
)


def create_sample_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "item_id": [10, 20, 30, 40],
            "feature_1": [1.0, 0.9, 0.0, 0.1],
            "feature_2": [0.0, 0.1, 1.0, 0.9],
        }
    )


def create_sample_interactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": [1, 2],
            "item_id": [10, 30],
            "events": [5, 4],
        }
    )


def test_content_based_can_fit() -> None:
    model = ContentBasedRecommender(
        item_features=create_sample_features()
    )

    model.fit(create_sample_interactions())

    assert model.is_fitted is True
    assert model.feature_matrix is not None
    assert 1 in model.user_profiles


def test_content_based_recommends_similar_unseen_item() -> None:
    model = ContentBasedRecommender(
        item_features=create_sample_features()
    )
    model.fit(create_sample_interactions())

    recommendations = model.recommend(user_id=1, k=1)

    assert recommendations == [20]


def test_content_based_excludes_seen_items() -> None:
    model = ContentBasedRecommender(
        item_features=create_sample_features()
    )
    model.fit(create_sample_interactions())

    recommendations = model.recommend(user_id=1, k=3)

    assert 10 not in recommendations


def test_unknown_user_returns_empty_list() -> None:
    model = ContentBasedRecommender(
        item_features=create_sample_features()
    )
    model.fit(create_sample_interactions())

    assert model.recommend(user_id=999, k=3) == []


def test_recommend_before_fit_raises_error() -> None:
    model = ContentBasedRecommender(
        item_features=create_sample_features()
    )

    with pytest.raises(RuntimeError):
        model.recommend(user_id=1)