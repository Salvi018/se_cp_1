# SDLC Model Selector — ML-based Cost & Effort Estimation

> Data-driven selection of SDLC models with ML-based cost & effort estimation.
> Built with React + Flask + Scikit-learn.

---

## Features

- **10 SDLC Models** — Waterfall, Agile, Scrum, Kanban, Spiral, Iterative, RAD, XP, SAFe, V-Model
- **ML Prediction** — Random Forest classifier (75%+ accuracy on 10 classes)
- **Cost & Effort Estimation** — Gradient Boosting regressors (R² > 0.90)
- **INR Currency** — All costs displayed in Indian Rupees (₹)
- **13 Input Features** — Including team experience, regulatory compliance, geographic distribution
- **Charts Page** — Feature importance + Predicted vs Actual scatter plots
- **Prediction History** — Paginated history of all past predictions
- **Quick Presets** — 6 project profiles (Enterprise, Startup, Medical, Prototype, Defence, R&D)
- **Single Server** — Flask serves both API and React frontend

---

## Quick Start

### Prerequisites
- Python 3.11+
- Node.js 20+

### 1. Clone
```bash
git clone https://github.com/Salvi018/se_cp_1.git
cd se_cp_1
```

### 2. Backend Setup
```bash
cd backend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # edit if needed
python3 app/ml/train.py       # trains models, saves .pkl files (~60s)
```

### 3. Frontend Build
```bash
cd frontend
npm install
npm run build                 # builds React into backend-served dist/
```

### 4. Run (single command)
```bash
cd backend
source venv/bin/activate
PORT=8080 python3 run.py
```

Open **http://localhost:8080**

---

## Project Structure

```
se_cp_1/
├── backend/
│   ├── app/
│   │   ├── ml/
│   │   │   ├── train.py          # ML training pipeline
│   │   │   ├── evaluate.py       # Standalone evaluation report
│   │   │   ├── data/             # Generated dataset (CSV)
│   │   │   └── models/           # Saved .pkl files (gitignored)
│   │   ├── routes/
│   │   │   ├── predict.py        # POST /api/predict
│   │   │   ├── history.py        # GET  /api/history
│   │   │   ├── metrics.py        # GET  /api/models/metrics
│   │   │   └── charts.py         # GET  /api/charts/*
│   │   ├── services/
│   │   │   ├── sdlc_classifier.py
│   │   │   └── effort_estimator.py
│   │   ├── models/db.py          # SQLAlchemy Prediction model
│   │   └── utils/preprocessor.py # Feature engineering + validation
│   ├── tests/
│   │   ├── conftest.py           # pytest fixtures
│   │   ├── test_api.py           # Integration tests (40+ tests)
│   │   └── test_models.py        # Model persistence tests
│   ├── .env.example
│   ├── build.sh                  # Render deploy script
│   ├── Procfile
│   ├── requirements.txt
│   └── run.py
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── ProjectForm.jsx   # Input form with presets
│   │   │   ├── ResultCard.jsx    # SDLC recommendation
│   │   │   ├── CostChart.jsx     # Cost breakdown bar chart
│   │   │   ├── HistoryTable.jsx  # Paginated history
│   │   │   ├── FeatureImportanceChart.jsx
│   │   │   ├── PredictedVsActualChart.jsx
│   │   │   └── LoadingSkeleton.jsx
│   │   ├── pages/
│   │   │   ├── Home.jsx
│   │   │   ├── Charts.jsx
│   │   │   └── History.jsx
│   │   ├── api/client.js         # Axios + interceptors
│   │   ├── hooks/usePrediction.js
│   │   └── utils/currency.js     # USD → INR conversion
│   └── .env.example
│
├── .github/workflows/ci.yml      # GitHub Actions CI/CD
└── README.md
```

---

## API Reference

### POST /api/predict

**Request Body:**
```json
{
  "team_size": 10,
  "duration_months": 12,
  "budget_usd": 200000,
  "requirements_clarity": 3,
  "client_involvement": 4,
  "tech_complexity": 3,
  "risk_level": "medium",
  "project_type": "web",
  "team_experience": 3,
  "regulatory_compliance": 0,
  "geographic_distribution": 1
}
```

| Field | Type | Range | Description |
|---|---|---|---|
| team_size | int | 1–100 | Number of team members |
| duration_months | int | 1–60 | Project duration |
| budget_usd | float | 1k–100M | Total budget in USD |
| requirements_clarity | int | 1–5 | 1=Vague, 5=Frozen |
| client_involvement | int | 1–5 | 1=Absent, 5=Daily |
| tech_complexity | int | 1–5 | 1=Simple, 5=Very Complex |
| risk_level | string | low/medium/high | Project risk |
| project_type | string | web/mobile/embedded/data | Type of project |
| team_experience | int | 1–5 | 1=Junior, 5=Expert |
| regulatory_compliance | int | 0/1 | HIPAA/ISO/GDPR required |
| geographic_distribution | int | 1–3 | 1=Co-located, 3=Distributed |

**Response:**
```json
{
  "success": true,
  "data": {
    "recommended_sdlc": "Scrum",
    "confidence": 0.82,
    "alternatives": ["Agile", "XP"],
    "effort_person_months": 68.5,
    "estimated_cost_usd": 242571,
    "cost_range": { "lower": 194057, "upper": 291086 }
  }
}
```

### GET /api/history?page=1&per_page=10
Returns paginated prediction history.

### GET /api/models/metrics
Returns ML model accuracy, MAE, RMSE, R², feature importances.

### GET /api/charts/feature-importance
Returns sorted feature importance array from Random Forest.

### GET /api/charts/predicted-vs-actual
Returns 120 sampled test points for cost and effort scatter plots.

---

## ML Models

| Model | Algorithm | Metric |
|---|---|---|
| SDLC Classifier | Random Forest (300 trees) | ~75% accuracy, F1 0.74 |
| Cost Estimator | Gradient Boosting (300 stages) | R² 0.78, MAPE 65% |
| Effort Estimator | Gradient Boosting (300 stages) | R² 0.92, MAPE 30% |

**13 Input Features:**
```
team_size, duration_months, budget_usd,
requirements_clarity, client_involvement, tech_complexity,
risk_encoded, type_encoded, budget_per_person, complexity_risk,
team_experience, regulatory_compliance, geographic_distribution
```

**Dataset:** 10,250 rows — synthetic + ISBSG/PROMISE-inspired real-world seeds with noise injection

---

## Running Tests

```bash
cd backend
source venv/bin/activate
pytest tests/ -v
```

```
tests/test_api.py      — 40+ integration tests (all endpoints)
tests/test_models.py   — model persistence & inference tests
```

---

## Environment Variables

### Backend (`backend/.env`)
```env
FLASK_ENV=production
SECRET_KEY=your-secret-key
PORT=8080
DATABASE_URL=sqlite:///app.db
```

### Frontend (`frontend/.env.production`)
```env
VITE_API_URL=https://your-backend.onrender.com/api
```

---

## Deployment

### Backend → Render
1. Connect GitHub repo at [render.com](https://render.com)
2. Root Directory: `backend`
3. Build Command: `./build.sh`
4. Start Command: `gunicorn run:app --config gunicorn.conf.py`
5. Add env vars: `SECRET_KEY`, `FLASK_ENV=production`

### Frontend → Netlify
1. Connect GitHub repo at [netlify.com](https://netlify.com)
2. Base directory: `frontend`, Build: `npm run build`, Publish: `dist`
3. Add env var: `VITE_API_URL=https://your-app.onrender.com/api`

### CI/CD
GitHub Actions runs automatically on every push to `main`:
- Trains ML models
- Runs all backend tests
- Builds frontend
- Deploys to Render (requires `RENDER_DEPLOY_HOOK_URL` secret)

---

## SDLC Model Decision Logic

| SDLC | When Recommended |
|---|---|
| Waterfall | Frozen reqs (clarity≥4), low risk |
| Agile | Balanced projects, medium everything |
| Scrum | High client involvement, evolving reqs |
| Kanban | Small team (≤5), short duration (≤6m), simple |
| Spiral | High risk + high complexity, safety-critical |
| Iterative | Vague reqs, low involvement, junior teams |
| RAD | Short duration (≤5m), prototype-driven, low risk |
| XP | Small expert team (≤6), TDD, high complexity |
| SAFe | Large enterprise (≥50 people), distributed |
| V-Model | Compliance required, medical/automotive/aerospace |
