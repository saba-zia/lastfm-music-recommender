from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "local"
    / "model_results.csv"
)

ASSETS_DIRECTORY = PROJECT_ROOT / "assets"

OUTPUT_FILE = ASSETS_DIRECTORY / "model_comparison.png"


def main() -> None:
    """Create a comparison chart for recommender performance."""

    if not RESULTS_FILE.exists():
        raise FileNotFoundError(
            f"Model results were not found: {RESULTS_FILE}"
        )

    results = pd.read_csv(RESULTS_FILE)

    required_columns = {
        "model",
        "metric",
        "k",
        "score",
    }

    missing_columns = required_columns.difference(results.columns)

    if missing_columns:
        raise ValueError(
            "The results file is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    results = results.sort_values(
        by="score",
        ascending=True,
    )

    ASSETS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axis = plt.subplots(
        figsize=(10, 6),
    )

    bars = axis.barh(
        results["model"],
        results["score"],
    )

    axis.set_title(
        "LastFM Recommender Model Comparison",
        fontsize=16,
    )

    axis.set_xlabel("HitRate@10")
    axis.set_ylabel("Model")

    axis.grid(
        axis="x",
        linestyle="--",
        alpha=0.4,
    )

    maximum_score = results["score"].max()

    axis.set_xlim(
        0,
        maximum_score * 1.2,
    )

    for bar, score in zip(
        bars,
        results["score"],
        strict=True,
    ):
        axis.text(
            bar.get_width() + maximum_score * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.4f}",
            va="center",
        )

    figure.tight_layout()

    figure.savefig(
        OUTPUT_FILE,
        dpi=300,
        bbox_inches="tight",
    )

    print(f"Chart saved to: {OUTPUT_FILE}")

    plt.show()


if __name__ == "__main__":
    main()