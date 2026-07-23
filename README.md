# 🎵 LastFM Music Recommender

A modular music recommendation system built in Python for implicit-feedback data, featuring Popularity, ItemKNN, Content-Based, ALS and Hybrid recommendation models.

[![Tests](https://github.com/saba-zia/lastfm-music-recommender/actions/workflows/tests.yml/badge.svg)](https://github.com/saba-zia/lastfm-music-recommender/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![Tests](https://img.shields.io/badge/tests-36%20passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)

> Built as an end-to-end machine learning portfolio project with modular model design, automated evaluation, visualization, unit testing and continuous integration.

---

## ⭐ Key Features

- Modular recommendation system built with Python
- Five recommendation approaches:
  - Popularity-based
  - ItemKNN collaborative filtering
  - Content-based filtering
  - Alternating Least Squares (ALS)
  - Hybrid recommendation
- Unified interface for training and recommendation
- Offline model evaluation
- Automated performance visualization
- 36 unit tests
- Continuous Integration with GitHub Actions
- Standard `src`-based Python package structure

---

## 🗺️ Project Roadmap

```mermaid
flowchart LR

    A[Project Setup] --> B[Data Loading]
    B --> C[Recommendation Models]
    C --> D[Offline Evaluation]
    D --> E[Visualization]
    E --> F[Testing and CI]
    F --> G[Documentation]
    G --> H[GitHub Portfolio]

    C --> P[Popularity]
    C --> I[ItemKNN]
    C --> CB[Content-Based]
    C --> ALS[ALS]
    C --> HY[Hybrid]
```

---

## ✅ Project Status

| Area | Status |
|------|--------|
| Project Structure | ✅ Completed |
| Data Loading and Validation | ✅ Completed |
| Popularity Recommender | ✅ Completed |
| ItemKNN Recommender | ✅ Completed |
| Content-Based Recommender | ✅ Completed |
| ALS Recommender | ✅ Completed |
| Hybrid Recommender | ✅ Completed |
| Offline Evaluation | ✅ Completed |
| Performance Visualization | ✅ Completed |
| Unit Tests | ✅ 36 Passing |
| GitHub Actions CI | ✅ Passing |
| Documentation | 🚧 In Progress |
| Interactive Demo | 📌 Planned |

---

## 🏗️ System Architecture

```mermaid
flowchart TD

    DATA[(LastFM Dataset)] --> LOADER[Data Loader]
    LOADER --> VALIDATION[Data Validation]

    VALIDATION --> MATRIX[User-Item Matrix]
    VALIDATION --> FEATURES[MusicNN Features]

    MATRIX --> POP[Popularity]
    MATRIX --> ITEM[ItemKNN]
    MATRIX --> ALS[ALS]

    FEATURES --> CONTENT[Content-Based]

    ITEM --> HYBRID[Hybrid]
    ALS --> HYBRID

    POP --> EVAL[Offline Evaluation]
    ITEM --> EVAL
    CONTENT --> EVAL
    ALS --> EVAL
    HYBRID --> EVAL

    EVAL --> RESULTS[(model_results.csv)]
    RESULTS --> CHART[Performance Visualization]
```

---

## 💻 Unified Model Interface

All recommendation models follow a consistent interface. This makes it easier to train, compare and replace models without changing the rest of the evaluation pipeline.

```python
model.fit(interactions)

recommendations = model.recommend(
    user_id=42,
    k=10,
)
```

---

## 🧪 Testing and Continuous Integration

The repository contains automated tests for the data-loading pipeline, evaluation functions and recommendation models.

Every push and pull request to the `main` branch automatically runs the test suite through GitHub Actions.

```bash
python -m pytest -v
```

Current test status:

```text
36 passed
```

---

## 🚀 Future Work

### Evaluation

- Precision@K
- Recall@K
- nDCG@K
- Mean Average Precision
- Recommendation coverage
- Recommendation diversity

### Recommendation Models

- Bayesian Personalized Ranking
- LightFM
- Neural Collaborative Filtering
- Graph-based recommendation

### Engineering

- Docker support
- Model persistence
- Configuration management
- Experiment tracking
- Code coverage reporting

### Deployment

- Streamlit web application
- FastAPI REST API
- Interactive recommendation dashboard
- Public cloud deployment