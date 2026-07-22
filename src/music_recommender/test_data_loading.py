from src.music_recommender.data import (
    load_interactions,
    load_items,
    load_users,
    load_musicnn_features,
    load_local_train,
    load_local_validation,
)


def main() -> None:
    interactions = load_interactions()
    items = load_items()
    users = load_users()
    musicnn = load_musicnn_features()
    local_train = load_local_train()
    local_validation = load_local_validation()

    print("Interactions:", interactions.shape)
    print("Items:", items.shape)
    print("Users:", users.shape)
    print("MusicNN:", musicnn.shape)
    print("Local train:", local_train.shape)
    print("Local validation:", local_validation.shape)

    print("\nSample interactions:")
    print(interactions.head())

    print("\nSample items:")
    print(items[["item_id", "artist", "song"]].head())


if __name__ == "__main__":
    main()