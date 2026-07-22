from statistics import mean

from music_recommender.config import TOP_K
from music_recommender.data import (
    load_local_train,
    load_local_validation,
    load_musicnn_features,
)
from music_recommender.evaluation import hit_rate_at_k
from music_recommender.models.content_based import (
    ContentBasedRecommender,
)


def main() -> None:
    """Train and evaluate the content-based recommender."""

    print("Loading local training data...")
    train_data = load_local_train()

    print("Loading local validation data...")
    validation_data = load_local_validation()

    print("Loading MusicNN item features...")
    musicnn_features = load_musicnn_features()

    print("Training content-based recommender...")
    model = ContentBasedRecommender(
        item_features=musicnn_features,
        user_column="user_id",
        item_column="item_id",
        weight_column="events",
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