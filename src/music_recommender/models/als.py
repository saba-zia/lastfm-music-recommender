from collections.abc import Hashable

import numpy as np
import pandas as pd
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix

from ..base import RecommenderModel


class ALSRecommender(RecommenderModel):
    """
    Alternating Least Squares recommender for implicit feedback.

    The model learns latent user and item representations from
    weighted user-item interaction data.
    """

    def __init__(
        self,
        factors: int = 64,
        iterations: int = 30,
        regularization: float = 0.05,
        alpha: float = 40.0,
        user_column: str = "user_id",
        item_column: str = "item_id",
        weight_column: str = "events",
        random_state: int = 42,
    ) -> None:
        """
        Initialize the ALS recommender.

        Parameters
        ----------
        factors:
            Number of latent dimensions used to represent users and items.
        iterations:
            Number of ALS optimization iterations.
        regularization:
            Regularization strength used to reduce overfitting.
        alpha:
            Scaling factor used to convert implicit interactions into
            confidence values.
        user_column:
            Name of the user identifier column.
        item_column:
            Name of the item identifier column.
        weight_column:
            Name of the interaction-strength column.
        random_state:
            Random seed used for reproducible model initialization.
        """
        if factors <= 0:
            raise ValueError("factors must be greater than zero.")

        if iterations <= 0:
            raise ValueError("iterations must be greater than zero.")

        if regularization < 0:
            raise ValueError("regularization must not be negative.")

        if alpha <= 0:
            raise ValueError("alpha must be greater than zero.")

        self.factors = factors
        self.iterations = iterations
        self.regularization = regularization
        self.alpha = alpha

        self.user_column = user_column
        self.item_column = item_column
        self.weight_column = weight_column
        self.random_state = random_state

        # Maps original IDs to continuous matrix indices.
        self.user_to_index: dict[Hashable, int] = {}
        self.item_to_index: dict[Hashable, int] = {}

        # Converts matrix item indices back to original item IDs.
        self.index_to_item: dict[int, Hashable] = {}

        self.model: AlternatingLeastSquares | None = None
        self.user_item_matrix: csr_matrix | None = None

        self.is_fitted = False

    def fit(self, interactions: pd.DataFrame) -> None:
        """
        Train the ALS model using implicit user-item interactions.

        Parameters
        ----------
        interactions:
            Dataframe containing user, item and interaction-weight columns.
        """
        self._validate_interactions(interactions)

        # Aggregate duplicate user-item interactions before building
        # the sparse matrix.
        aggregated_interactions = (
            interactions.groupby(
                [self.user_column, self.item_column],
                as_index=False,
            )[self.weight_column]
            .sum()
        )

        unique_users = (
            aggregated_interactions[self.user_column]
            .drop_duplicates()
            .tolist()
        )

        unique_items = (
            aggregated_interactions[self.item_column]
            .drop_duplicates()
            .tolist()
        )

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

        row_indices = (
            aggregated_interactions[self.user_column]
            .map(self.user_to_index)
            .to_numpy()
        )

        column_indices = (
            aggregated_interactions[self.item_column]
            .map(self.item_to_index)
            .to_numpy()
        )

        interaction_weights = (
            aggregated_interactions[self.weight_column]
            .astype(float)
            .to_numpy()
        )

        # In implicit-feedback ALS, interaction values represent confidence,
        # not explicit ratings.
        confidence_values = interaction_weights * self.alpha

        self.user_item_matrix = csr_matrix(
            (
                confidence_values,
                (row_indices, column_indices),
            ),
            shape=(
                len(self.user_to_index),
                len(self.item_to_index),
            ),
            dtype=np.float32,
        )

        self.model = AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            random_state=self.random_state,
        )

        self.model.fit(self.user_item_matrix)

        self.is_fitted = True

    def recommend(
        self,
        user_id: int,
        k: int = 10,
    ) -> list[int]:
        """
        Generate top-k unseen item recommendations for one user.

        Parameters
        ----------
        user_id:
            Original identifier of the target user.
        k:
            Maximum number of recommendations.

        Returns
        -------
        list[int]
            Ranked original item identifiers.

        Raises
        ------
        RuntimeError
            If the model has not been trained.
        ValueError
            If k is not positive.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "The model has not been fitted. Call fit() before recommend()."
            )

        if k <= 0:
            raise ValueError("k must be greater than zero.")

        # ALS cannot construct a latent profile for a completely unknown user.
        if user_id not in self.user_to_index:
            return []

        if self.model is None or self.user_item_matrix is None:
            raise RuntimeError("Model state is incomplete.")

        user_index = self.user_to_index[user_id]
        user_interactions = self.user_item_matrix.getrow(user_index)

        item_indices, _ = self.model.recommend(
            userid=user_index,
            user_items=user_interactions,
            N=k,
            filter_already_liked_items=True,
        )

        recommendations: list[int] = []

        for item_index in np.asarray(item_indices).ravel():
            original_item_id = self.index_to_item.get(int(item_index))

            if original_item_id is not None:
                recommendations.append(int(original_item_id))

        return recommendations

    def _validate_interactions(
        self,
        interactions: pd.DataFrame,
    ) -> None:
        """
        Validate interaction data before model training.

        Parameters
        ----------
        interactions:
            Dataframe to validate.

        Raises
        ------
        TypeError
            If interactions is not a pandas DataFrame.
        ValueError
            If the dataframe is empty, required columns are missing,
            or interaction values are invalid.
        """
        if not isinstance(interactions, pd.DataFrame):
            raise TypeError("interactions must be a pandas DataFrame.")

        if interactions.empty:
            raise ValueError(
                "The interaction dataframe must not be empty."
            )

        required_columns = {
            self.user_column,
            self.item_column,
            self.weight_column,
        }

        missing_columns = required_columns.difference(
            interactions.columns
        )

        if missing_columns:
            raise ValueError(
                "The interaction dataframe is missing required columns: "
                f"{sorted(missing_columns)}"
            )

        if interactions[self.user_column].isna().any():
            raise ValueError(
                f"Column '{self.user_column}' contains missing values."
            )

        if interactions[self.item_column].isna().any():
            raise ValueError(
                f"Column '{self.item_column}' contains missing values."
            )

        numeric_weights = pd.to_numeric(
            interactions[self.weight_column],
            errors="coerce",
        )

        if numeric_weights.isna().any():
            raise ValueError(
                f"Column '{self.weight_column}' must contain only "
                "valid numeric values."
            )

        if (numeric_weights < 0).any():
            raise ValueError(
                f"Column '{self.weight_column}' must not contain "
                "negative values."
            )