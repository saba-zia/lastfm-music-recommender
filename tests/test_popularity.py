import pandas as pd

from src.music_recommender.models.popularity import (
    PopularityRecommender,
)


def create_sample_interactions() -> pd.DataFrame:
    """Create a small interaction dataset for testing."""
    return pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 3],
            "item_id": [10, 20, 10, 30, 10],
            "events": [5, 2, 3, 4, 1],
        }
    )


def test_popularity_ranking() -> None:
    interactions = create_sample_interactions()

    model = PopularityRecommender()
    model.fit(interactions)

    assert model.popular_items == [10, 30, 20]


def test_seen_items_are_excluded() -> None:
    interactions = create_sample_interactions()

    model = PopularityRecommender()
    model.fit(interactions)

    recommendations = model.recommend(user_id=1, k=2)

    assert recommendations == [30]


def test_unknown_user_receives_popular_items() -> None:
    interactions = create_sample_interactions()

    model = PopularityRecommender()
    model.fit(interactions)

    recommendations = model.recommend(user_id=999, k=2)

    assert recommendations == [10, 30]


def test_recommend_before_fit_raises_error() -> None:
    model = PopularityRecommender()

    try:
        model.recommend(user_id=1)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Expected RuntimeError was not raised.")