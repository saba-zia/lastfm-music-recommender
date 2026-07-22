import pandas as pd
import pytest

from music_recommender.models.als import ALSRecommender


def create_sample_interactions() -> pd.DataFrame:
    """
    Create a small implicit-feedback dataset for ALS testing.
    """
    return pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 3, 3],
            "item_id": [10, 20, 10, 30, 20, 40],
            "events": [5, 2, 4, 3, 4, 2],
        }
    )


def create_test_model() -> ALSRecommender:
    """
    Create a small and fast ALS model for unit tests.
    """
    return ALSRecommender(
        factors=4,
        iterations=3,
        regularization=0.1,
        alpha=10.0,
        random_state=42,
    )


def test_als_initializes_with_valid_parameters() -> None:
    model = ALSRecommender(
        factors=32,
        iterations=10,
        regularization=0.1,
        alpha=20.0,
    )

    assert model.factors == 32
    assert model.iterations == 10
    assert model.regularization == 0.1
    assert model.alpha == 20.0
    assert model.is_fitted is False


def test_invalid_factors_raise_error() -> None:
    with pytest.raises(
        ValueError,
        match="factors must be greater than zero",
    ):
        ALSRecommender(factors=0)


def test_invalid_iterations_raise_error() -> None:
    with pytest.raises(
        ValueError,
        match="iterations must be greater than zero",
    ):
        ALSRecommender(iterations=0)


def test_negative_regularization_raises_error() -> None:
    with pytest.raises(
        ValueError,
        match="regularization must not be negative",
    ):
        ALSRecommender(regularization=-0.1)


def test_invalid_alpha_raises_error() -> None:
    with pytest.raises(
        ValueError,
        match="alpha must be greater than zero",
    ):
        ALSRecommender(alpha=0)


def test_als_can_fit() -> None:
    model = create_test_model()

    model.fit(create_sample_interactions())

    assert model.is_fitted is True
    assert model.model is not None
    assert model.user_item_matrix is not None

    assert len(model.user_to_index) == 3
    assert len(model.item_to_index) == 4
    assert len(model.index_to_item) == 4

    assert model.user_item_matrix.shape == (3, 4)


def test_als_excludes_seen_items() -> None:
    model = create_test_model()
    model.fit(create_sample_interactions())

    recommendations = model.recommend(
        user_id=1,
        k=2,
    )

    assert 10 not in recommendations
    assert 20 not in recommendations


def test_als_unknown_user_returns_empty_list() -> None:
    model = create_test_model()
    model.fit(create_sample_interactions())

    recommendations = model.recommend(
        user_id=999,
        k=2,
    )

    assert recommendations == []


def test_als_recommend_before_fit_raises_error() -> None:
    model = create_test_model()

    with pytest.raises(
        RuntimeError,
        match="The model has not been fitted",
    ):
        model.recommend(user_id=1)


def test_als_rejects_non_positive_k() -> None:
    model = create_test_model()
    model.fit(create_sample_interactions())

    with pytest.raises(
        ValueError,
        match="k must be greater than zero",
    ):
        model.recommend(
            user_id=1,
            k=0,
        )


def test_als_rejects_empty_interactions() -> None:
    model = create_test_model()

    empty_interactions = pd.DataFrame(
        columns=["user_id", "item_id", "events"]
    )

    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        model.fit(empty_interactions)


def test_als_rejects_missing_columns() -> None:
    model = create_test_model()

    invalid_interactions = pd.DataFrame(
        {
            "user_id": [1],
            "item_id": [10],
        }
    )

    with pytest.raises(
        ValueError,
        match="missing required columns",
    ):
        model.fit(invalid_interactions)


def test_als_rejects_negative_interaction_values() -> None:
    model = create_test_model()

    invalid_interactions = pd.DataFrame(
        {
            "user_id": [1],
            "item_id": [10],
            "events": [-1],
        }
    )

    with pytest.raises(
        ValueError,
        match="must not contain negative values",
    ):
        model.fit(invalid_interactions)