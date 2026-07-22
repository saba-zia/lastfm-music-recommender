import pandas as pd
import pytest

from music_recommender.models.item_knn import ItemKNNRecommender


def create_sample_interactions() -> pd.DataFrame:
    """
    Create a small dataset with predictable item similarity.
    """
    return pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 3, 3],
            "item_id": [10, 20, 10, 30, 20, 40],
            "events": [1, 1, 1, 1, 1, 1],
        }
    )


def test_item_knn_can_fit() -> None:
    model = ItemKNNRecommender(n_neighbors=2)

    model.fit(create_sample_interactions())

    assert model.is_fitted is True
    assert model.user_item_matrix is not None
    assert model.item_similarity is not None


def test_item_knn_excludes_seen_items() -> None:
    model = ItemKNNRecommender(n_neighbors=3)
    model.fit(create_sample_interactions())

    recommendations = model.recommend(user_id=1, k=3)

    assert 10 not in recommendations
    assert 20 not in recommendations


def test_unknown_user_returns_empty_list() -> None:
    model = ItemKNNRecommender(n_neighbors=2)
    model.fit(create_sample_interactions())

    recommendations = model.recommend(user_id=999, k=3)

    assert recommendations == []


def test_recommend_before_fit_raises_error() -> None:
    model = ItemKNNRecommender()

    with pytest.raises(RuntimeError):
        model.recommend(user_id=1)


def test_invalid_neighbor_count_raises_error() -> None:
    with pytest.raises(ValueError):
        ItemKNNRecommender(n_neighbors=0)