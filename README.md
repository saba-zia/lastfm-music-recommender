## 🗺️ Project Roadmap

```mermaid
flowchart TD

A[Project Setup]
B[Data Loading]
C[Recommendation Models]
D[Offline Evaluation]
E[Visualization]
F[Documentation]
G[GitHub Portfolio]

A --> B
B --> C
C --> D
D --> E
E --> F
F --> G

C --> P[Popularity]
C --> I[ItemKNN]
C --> CB[Content-Based]
C --> ALS[ALS]
C --> H[Hybrid]
```

### Development Status

| Component | Status |
|------------|--------|
| Project Structure | ✅ Completed |
| Data Loading | ✅ Completed |
| Evaluation Framework | ✅ Completed |
| Popularity Model | ✅ Completed |
| ItemKNN | ✅ Completed |
| Content-Based | ✅ Completed |
| ALS | ✅ Completed |
| Hybrid | ✅ Completed |
| Visualization | ✅ Completed |
| Unit Tests | ✅ 36 Passed |
| Documentation | 🟡 In Progress |
| GitHub Actions | ⏳ Planned |
| Streamlit Demo | ⏳ Planned |

---

## 🏗️ System Architecture

```mermaid
flowchart TD

DATA[(LastFM Dataset)]

DATA --> LOADER[Data Loader]

LOADER --> VALIDATION[Validation]

VALIDATION --> MATRIX[User-Item Matrix]

VALIDATION --> FEATURES[MusicNN Features]

MATRIX --> POP[Popularity]

MATRIX --> ITEM[ItemKNN]

MATRIX --> ALS[ALS]

FEATURES --> CONTENT[Content-Based]

ALS --> HYBRID[Hybrid]

ITEM --> HYBRID

POP --> EVAL[Offline Evaluation]
ITEM --> EVAL
CONTENT --> EVAL
ALS --> EVAL
HYBRID --> EVAL

EVAL --> RESULTS[(model_results.csv)]

RESULTS --> CHART[Performance Visualization]
```

Every recommender implements the same interface:

```python
model.fit(interactions)
recommendations = model.recommend(user_id, k=10)
```

---

## 🔮 Future Work

### Evaluation

- Precision@K
- Recall@K
- nDCG@K
- Mean Average Precision (MAP)

### Recommendation Models

- Bayesian Personalized Ranking (BPR)
- LightFM
- Neural Collaborative Filtering
- Graph-based Recommendation

### Engineering

- GitHub Actions (Continuous Integration)
- Docker support
- Model persistence
- Configuration files
- Experiment tracking

### Deployment

- Streamlit Web Application
- FastAPI REST API
- Public cloud deployment
- Interactive recommendation dashboard