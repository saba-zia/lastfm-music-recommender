from statistics import mean

from music_recommender.config import TOP_K
from music_recommender.data import (
    load_local_train,
    load_local_validation,
)
from music_recommender.evaluation import hit_rate_at_k
from music_recommender.models.als import ALSRecommender
from music_recommender.models.hybrid import HybridRecommender
from music_recommender.models.item_knn import ItemKNNRecommender


def main() -> None:
    """Train and evaluate the Hybrid recommender."""

    print("Loading local training data...")
    train_data = load_local_train()

    print("Loading local validation data...")
    validation_data = load_local_validation()

    print("Building component models...")

    als_model = ALSRecommender(
        factors=64,
        iterations=30,
        regularization=0.05,
        alpha=40.0,
        user_column="user_id",
        item_column="item_id",
        weight_column="events",
        random_state=42,
    )

    item_knn_model = ItemKNNRecommender(
        n_neighbors=100,
        user_column="user_id",
        item_column="item_id",
        weight_column="events",
    )

    print("Training Hybrid recommender...")

    model = HybridRecommender(
        primary_model=als_model,
        secondary_model=item_knn_model,
        primary_weight=0.7,
        secondary_weight=0.3,
        candidate_multiplier=5,
    )

    model.fit(train_data)

    relevant_items_by_user = (
        validation_data.groupby("user_id")["item_id"]
        .apply(set)
        .to_dict()
    )

    user_scores: list[float] = []

    print(f"Evaluating HitRate@{TOP_K}...")

    for user_id, relevant_items in relevant_items_by_user.items():
        recommendations = model.recommend(
            user_id=user_id,
            k=TOP_K,
        )

        score = hit_rate_at_k(
            recommendations=recommendations,
            relevant_items=relevant_items,
            k=TOP_K,
        )

        user_scores.append(score)

    final_hit_rate = mean(user_scores) if user_scores else 0.0

    print()
    print("Evaluation completed.")
    print(f"Validation users: {len(user_scores)}")
    print(f"HitRate@{TOP_K}: {final_hit_rate:.6f}")


if __name__ == "__main__":
    main()