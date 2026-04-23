# SDLC Model Selector — ML-based Cost & Effort Estimation

## Quick Start

### Backend
```bash
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python app/ml/train.py      # trains models, saves .pkl files
python run.py               # starts Flask on http://localhost:5000
```

### Frontend
```bash
cd frontend
npm install
npm run dev                 # starts Vite on http://localhost:5173
```

## API Endpoints

| Method | Endpoint             | Description                        |
|--------|----------------------|------------------------------------|
| POST   | /api/predict         | Predict SDLC + estimate cost       |
| GET    | /api/history         | Paginated prediction history       |
| GET    | /api/models/metrics  | ML model accuracy & MAE            |

## POST /api/predict — Request Body
```json
{
  "team_size": 10,
  "duration_months": 12,
  "budget_usd": 100000,
  "requirements_clarity": 3,
  "client_involvement": 4,
  "tech_complexity": 3,
  "risk_level": "medium",
  "project_type": "web"
}
```

## POST /api/predict — Response
```json
{
  "recommended_sdlc": "Agile",
  "confidence": 0.87,
  "alternatives": ["Scrum", "Kanban"],
  "effort_person_months": 7.2,
  "estimated_cost_usd": 84200,
  "cost_lower": 67360,
  "cost_upper": 101040
}
```

## ML Models
- SDLC Classifier: Random Forest (200 estimators)
- Cost Estimator: Gradient Boosting Regressor (200 estimators)
- Features: team size, duration, budget, clarity, involvement, complexity, risk, type + 2 derived features
