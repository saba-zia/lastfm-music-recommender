from collections.abc import Hashable

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from ..base import RecommenderModel


class ItemKNNRecommender(RecommenderModel):
    """
    Item-based collaborative filtering recommender.

    The model computes cosine similarity between items using
    user-item interaction data.
    """

    def __init__(
        self,
        n_neighbors: int = 100,
        user_column: str = "user_id",
        item_column: str = "item_id",
        weight_column: str = "events",
    ) -> None:
        """
        Initialize the ItemKNN recommender.

        Parameters
        ----------
        n_neighbors:
            Maximum number of similar items used for each item.
        user_column:
            Name of the user identifier column.
        item_column:
            Name of the item identifier column.
        weight_column:
            Name of the interaction weight column.
        """
        if n_neighbors <= 0:
            raise ValueError("n_neighbors must be greater than zero.")

        self.n_neighbors = n_neighbors
        self.user_column = user_column
        self.item_column = item_column
        self.weight_column = weight_column

        self.user_to_index: dict[Hashable, int] = {}
        self.item_to_index: dict[Hashable, int] = {}
        self.index_to_item: dict[int, Hashable] = {}

        self.user_item_matrix: csr_matrix | None = None
        self.item_similarity: np.ndarray | None = None

        self.is_fitted = False

    def fit(self, interactions: pd.DataFrame) -> None:
        """
        Build the user-item matrix and compute item similarities.
        """
        self._validate_interactions(interactions)

        unique_users = interactions[self.user_column].drop_duplicates().tolist()
        unique_items = interactions[self.item_column].drop_duplicates().tolist()

        self.user_to_index = {
            user_id: index
            for index, user_id in enumerate(unique_users)
        }

        self.item_to_index = {
            item_id: index
            for index, item_id in enumerate(unique_items)
        }

        self.index_to_item = {
            index: item_id
            for item_id, index in self.item_to_index.items()
        }

        row_indices = interactions[self.user_column].map(
            self.user_to_index
        ).to_numpy()

        column_indices = interactions[self.item_column].map(
            self.item_to_index
        ).to_numpy()

        values = interactions[self.weight_column].astype(float).to_numpy()

        self.user_item_matrix = csr_matrix(
            (
                values,
                (row_indices, column_indices),
            ),
            shape=(
                len(self.user_to_index),
                len(self.item_to_index),
            ),
        )

        item_user_matrix = self.user_item_matrix.T

        similarity_matrix = cosine_similarity(
            item_user_matrix,
            dense_output=True,
        )

        np.fill_diagonal(similarity_matrix, 0.0)

        if self.n_neighbors < similarity_matrix.shape[1]:
            neighbor_indices = np.argpartition(
                similarity_matrix,
                -self.n_neighbors,
                axis=1,
            )[:, -self.n_neighbors:]

            sparse_similarity = np.zeros_like(similarity_matrix)

            row_indices_matrix = np.arange(
                similarity_matrix.shape[0]
            )[:, None]

            sparse_similarity[
                row_indices_matrix,
                neighbor_indices,
            ] = similarity_matrix[
                row_indices_matrix,
                neighbor_indices,
            ]

            similarity_matrix = sparse_similarity

        self.item_similarity = similarity_matrix
        self.is_fitted = True

    def recommend(
        self,
        user_id: int,
        k: int = 10,
    ) -> list[int]:
        """
        Generate top-k recommendations for one user.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "The model has not been fitted. Call fit() before recommend()."
            )

        if k <= 0:
            raise ValueError("k must be greater than zero.")

        if user_id not in self.user_to_index:
            return []

        if self.user_item_matrix is None or self.item_similarity is None:
            raise RuntimeError("Model state is incomplete.")

        user_index = self.user_to_index[user_id]

        user_vector = self.user_item_matrix.getrow(user_index)

        item_scores = user_vector @ self.item_similarity
        item_scores = np.asarray(item_scores).ravel()

        seen_item_indices = user_vector.indices
        item_scores[seen_item_indices] = -np.inf

        ranked_indices = np.argsort(-item_scores)

        recommendations: list[int] = []

        for item_index in ranked_indices:
            if not np.isfinite(item_scores[item_index]):
                continue

            if item_scores[item_index] <= 0:
                continue

            recommendations.append(
                self.index_to_item[item_index]
            )

            if len(recommendations) == k:
                break

        return recommendations

    def _validate_interactions(
        self,
        interactions: pd.DataFrame,
    ) -> None:
        """
        Validate the interaction dataframe.
        """
        if interactions.empty:
            raise ValueError("The interaction dataframe must not be empty.")

        required_columns = {
            self.user_column,
            self.item_column,
            self.weight_column,
        }

        missing_columns = required_columns.difference(interactions.columns)

        if missing_columns:
            raise ValueError(
                "The interaction dataframe is missing required columns: "
                f"{sorted(missing_columns)}"
            )

        if interactions[self.weight_column].isna().any():
            raise ValueError(
                f"Column '{self.weight_column}' contains missing values."
            )

        if (interactions[self.weight_column] < 0).any():
            raise ValueError(
                f"Column '{self.weight_column}' must not contain "
                "negative values."
            )