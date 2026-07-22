from collections.abc import Hashable

import pandas as pd

from ..base import RecommenderModel


class HybridRecommender(RecommenderModel):
    """
    Combine two recommender models using weighted rank fusion.

    Each component model generates a ranked candidate list. Items receive
    higher scores when they appear near the top of either list.
    """

    def __init__(
        self,
        primary_model: RecommenderModel,
        secondary_model: RecommenderModel,
        primary_weight: float = 0.7,
        secondary_weight: float = 0.3,
        candidate_multiplier: int = 5,
    ) -> None:
        """
        Initialize the hybrid recommender.

        Parameters
        ----------
        primary_model:
            Main recommender, typically ALS.
        secondary_model:
            Supporting recommender, typically ItemKNN.
        primary_weight:
            Weight assigned to rankings from the primary model.
        secondary_weight:
            Weight assigned to rankings from the secondary model.
        candidate_multiplier:
            Number of candidates requested from each model relative to k.
        """
        if primary_weight < 0:
            raise ValueError("primary_weight must not be negative.")

        if secondary_weight < 0:
            raise ValueError("secondary_weight must not be negative.")

        if primary_weight + secondary_weight <= 0:
            raise ValueError(
                "At least one hybrid weight must be greater than zero."
            )

        if candidate_multiplier <= 0:
            raise ValueError(
                "candidate_multiplier must be greater than zero."
            )

        self.primary_model = primary_model
        self.secondary_model = secondary_model

        total_weight = primary_weight + secondary_weight

        self.primary_weight = primary_weight / total_weight
        self.secondary_weight = secondary_weight / total_weight
        self.candidate_multiplier = candidate_multiplier

        self.is_fitted = False

    def fit(self, interactions: pd.DataFrame) -> None:
        """
        Train both component recommenders.

        Parameters
        ----------
        interactions:
            User-item interaction dataframe.
        """
        self.primary_model.fit(interactions)
        self.secondary_model.fit(interactions)

        self.is_fitted = True

    def recommend(
        self,
        user_id: int,
        k: int = 10,
    ) -> list[int]:
        """
        Generate top-k recommendations using weighted rank fusion.

        Parameters
        ----------
        user_id:
            Identifier of the target user.
        k:
            Number of recommendations.

        Returns
        -------
        list[int]
            Ranked hybrid recommendations.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "The model has not been fitted. Call fit() before recommend()."
            )

        if k <= 0:
            raise ValueError("k must be greater than zero.")

        candidate_count = k * self.candidate_multiplier

        primary_recommendations = self.primary_model.recommend(
            user_id=user_id,
            k=candidate_count,
        )

        secondary_recommendations = self.secondary_model.recommend(
            user_id=user_id,
            k=candidate_count,
        )

        item_scores: dict[Hashable, float] = {}

        self._add_rank_scores(
            recommendations=primary_recommendations,
            weight=self.primary_weight,
            scores=item_scores,
        )

        self._add_rank_scores(
            recommendations=secondary_recommendations,
            weight=self.secondary_weight,
            scores=item_scores,
        )

        ranked_items = sorted(
            item_scores,
            key=lambda item_id: (
                -item_scores[item_id],
                str(item_id),
            ),
        )

        return [
            int(item_id)
            for item_id in ranked_items[:k]
        ]

    @staticmethod
    def _add_rank_scores(
        recommendations: list[int],
        weight: float,
        scores: dict[Hashable, float],
    ) -> None:
        """
        Add reciprocal-rank scores from one model.

        Higher-ranked items receive larger scores.
        """
        for rank, item_id in enumerate(recommendations, start=1):
            rank_score = weight / rank

            scores[item_id] = (
                scores.get(item_id, 0.0) + rank_score
            )