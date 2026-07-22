from abc import ABC, abstractmethod

import pandas as pd


class RecommenderModel(ABC):
    """
    Abstract base class for all recommender models.

    Every recommender implementation must provide:
    - fit
    - recommend
    """

    @abstractmethod
    def fit(self, interactions: pd.DataFrame) -> None:
        """
        Train the recommender model.

        Parameters
        ----------
        interactions:
            User-item interaction dataframe.
        """
        raise NotImplementedError

    @abstractmethod
    def recommend(
        self,
        user_id: int,
        k: int = 10,
    ) -> list[int]:
        """
        Generate top-k item recommendations for a user.

        Parameters
        ----------
        user_id:
            Identifier of the target user.
        k:
            Number of recommendations.

        Returns
        -------
        list[int]
            Ranked item identifiers.
        """
        raise NotImplementedError