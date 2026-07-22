from collections.abc import Hashable

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

from ..base import RecommenderModel


class ContentBasedRecommender(RecommenderModel):
    """
    Recommend items using MusicNN item features.

    A user profile is created by averaging the feature vectors of
    items previously consumed by that user. Recommendations are ranked
    by cosine similarity between the user profile and item features.
    """

    def __init__(
        self,
        item_features: pd.DataFrame,
        user_column: str = "user_id",
        item_column: str = "item_id",
        weight_column: str = "events",
    ) -> None:
        """
        Initialize the content-based recommender.

        Parameters
        ----------
        item_features:
            Dataframe containing item_id and numerical item features.
        user_column:
            Name of the user identifier column.
        item_column:
            Name of the item identifier column.
        weight_column:
            Name of the interaction-strength column.
        """
        self.user_column = user_column
        self.item_column = item_column
        self.weight_column = weight_column

        self.item_features = item_features.copy()

        self.item_to_index: dict[Hashable, int] = {}
        self.index_to_item: dict[int, Hashable] = {}

        self.feature_matrix: np.ndarray | None = None
        self.user_profiles: dict[Hashable, np.ndarray] = {}
        self.user_seen_items: dict[Hashable, set[Hashable]] = {}

        self.is_fitted = False

    def fit(self, interactions: pd.DataFrame) -> None:
        """
        Build normalized item vectors and user preference profiles.

        Parameters
        ----------
        interactions:
            User-item interaction dataframe.
        """
        self._validate_inputs(interactions)

        feature_columns = [
            column
            for column in self.item_features.columns
            if column != self.item_column
        ]

        available_features = self.item_features.dropna(
            subset=[self.item_column]
        ).copy()

        available_features[feature_columns] = available_features[
            feature_columns
        ].apply(pd.to_numeric, errors="coerce")

        available_features = available_features.dropna(
            subset=feature_columns
        )

        item_ids = available_features[self.item_column].tolist()

        self.item_to_index = {
            item_id: index
            for index, item_id in enumerate(item_ids)
        }

        self.index_to_item = {
            index: item_id
            for item_id, index in self.item_to_index.items()
        }

        raw_feature_matrix = available_features[
            feature_columns
        ].to_numpy(dtype=float)

        self.feature_matrix = normalize(
            raw_feature_matrix,
            norm="l2",
            axis=1,
        )

        self.user_seen_items = (
            interactions.groupby(self.user_column)[self.item_column]
            .apply(set)
            .to_dict()
        )

        for user_id, user_interactions in interactions.groupby(
            self.user_column
        ):
            valid_interactions = user_interactions[
                user_interactions[self.item_column].isin(self.item_to_index)
            ]

            if valid_interactions.empty:
                continue

            item_indices = (
                valid_interactions[self.item_column]
                .map(self.item_to_index)
                .to_numpy()
            )

            weights = (
                valid_interactions[self.weight_column]
                .astype(float)
                .to_numpy()
            )

            user_item_vectors = self.feature_matrix[item_indices]

            if weights.sum() > 0:
                user_profile = np.average(
                    user_item_vectors,
                    axis=0,
                    weights=weights,
                )
            else:
                user_profile = user_item_vectors.mean(axis=0)

            profile_norm = np.linalg.norm(user_profile)

            if profile_norm > 0:
                user_profile = user_profile / profile_norm

            self.user_profiles[user_id] = user_profile

        self.is_fitted = True

    def recommend(
        self,
        user_id: int,
        k: int = 10,
    ) -> list[int]:
        """
        Generate top-k content-based recommendations.

        Parameters
        ----------
        user_id:
            Target user identifier.
        k:
            Maximum number of recommendations.

        Returns
        -------
        list[int]
            Ranked unseen item identifiers.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "The model has not been fitted. Call fit() before recommend()."
            )

        if k <= 0:
            raise ValueError("k must be greater than zero.")

        if self.feature_matrix is None:
            raise RuntimeError("Feature matrix is unavailable.")

        if user_id not in self.user_profiles:
            return []

        user_profile = self.user_profiles[user_id]

        item_scores = self.feature_matrix @ user_profile

        seen_items = self.user_seen_items.get(user_id, set())

        for item_id in seen_items:
            item_index = self.item_to_index.get(item_id)

            if item_index is not None:
                item_scores[item_index] = -np.inf

        ranked_indices = np.argsort(-item_scores)

        recommendations: list[int] = []

        for item_index in ranked_indices:
            if not np.isfinite(item_scores[item_index]):
                continue

            recommendations.append(
                self.index_to_item[item_index]
            )

            if len(recommendations) == k:
                break

        return recommendations

    def _validate_inputs(
        self,
        interactions: pd.DataFrame,
    ) -> None:
        """
        Validate interaction data and item features.
        """
        if interactions.empty:
            raise ValueError("The interaction dataframe must not be empty.")

        if self.item_features.empty:
            raise ValueError("The item feature dataframe must not be empty.")

        required_interaction_columns = {
            self.user_column,
            self.item_column,
            self.weight_column,
        }

        missing_interaction_columns = (
            required_interaction_columns.difference(interactions.columns)
        )

        if missing_interaction_columns:
            raise ValueError(
                "The interaction dataframe is missing required columns: "
                f"{sorted(missing_interaction_columns)}"
            )

        if self.item_column not in self.item_features.columns:
            raise ValueError(
                f"Item features must contain '{self.item_column}'."
            )

        feature_columns = [
            column
            for column in self.item_features.columns
            if column != self.item_column
        ]

        if not feature_columns:
            raise ValueError(
                "The item feature dataframe must contain feature columns."
            )