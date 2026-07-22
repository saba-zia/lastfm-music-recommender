from collections.abc import Hashable

import pandas as pd

from ..base import RecommenderModel


class PopularityRecommender(RecommenderModel):
    """
    Recommend globally popular items.

    The model ranks items by the total interaction weight they receive
    in the training data. Items already seen by a user are excluded from
    that user's recommendations.
    """

    def __init__(
        self,
        user_column: str = "user_id",
        item_column: str = "item_id",
        weight_column: str = "events",
    ) -> None:
        """
        Initialize the popularity recommender.

        Parameters
        ----------
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

        self.popular_items: list[Hashable] = []
        self.user_seen_items: dict[Hashable, set[Hashable]] = {}
        self.is_fitted = False

    def fit(self, interactions: pd.DataFrame) -> None:
        """
        Learn the global popularity ranking from interaction data.

        Parameters
        ----------
        interactions:
            Dataframe containing user, item and interaction-weight columns.

        Raises
        ------
        ValueError
            If required columns are missing or the dataframe is empty.
        """
        self._validate_interactions(interactions)

        item_scores = (
            interactions.groupby(self.item_column, as_index=False)[
                self.weight_column
            ]
            .sum()
            .sort_values(
                by=[self.weight_column, self.item_column],
                ascending=[False, True],
            )
        )

        self.popular_items = item_scores[self.item_column].tolist()

        self.user_seen_items = (
            interactions.groupby(self.user_column)[self.item_column]
            .apply(set)
            .to_dict()
        )

        self.is_fitted = True

    def recommend(
        self,
        user_id: int,
        k: int = 10,
    ) -> list[int]:
        """
        Generate top-k unseen popular-item recommendations.

        Parameters
        ----------
        user_id:
            Identifier of the target user.
        k:
            Maximum number of recommendations.

        Returns
        -------
        list[int]
            Ranked unseen item identifiers.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        ValueError
            If k is not positive.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "The model has not been fitted. Call fit() before recommend()."
            )

        if k <= 0:
            raise ValueError("k must be greater than zero.")

        seen_items = self.user_seen_items.get(user_id, set())

        recommendations = [
            item_id
            for item_id in self.popular_items
            if item_id not in seen_items
        ]

        return recommendations[:k]

    def _validate_interactions(
        self,
        interactions: pd.DataFrame,
    ) -> None:
        """
        Validate the interaction dataframe before training.
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