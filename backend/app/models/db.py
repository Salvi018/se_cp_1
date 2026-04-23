from datetime import datetime
from app import db

class Prediction(db.Model):
    __tablename__ = "predictions"

    id                    = db.Column(db.Integer, primary_key=True)
    created_at            = db.Column(db.DateTime, default=datetime.utcnow)

    # Inputs
    team_size             = db.Column(db.Integer)
    duration_months       = db.Column(db.Integer)
    budget_usd            = db.Column(db.Float)
    requirements_clarity  = db.Column(db.Integer)
    client_involvement    = db.Column(db.Integer)
    tech_complexity       = db.Column(db.Integer)
    risk_level            = db.Column(db.String(10))
    project_type          = db.Column(db.String(20))

    # Outputs
    recommended_sdlc      = db.Column(db.String(30))
    confidence            = db.Column(db.Float)
    alternatives          = db.Column(db.String(100))
    effort_person_months  = db.Column(db.Float)
    estimated_cost_usd    = db.Column(db.Float)
    cost_lower            = db.Column(db.Float)
    cost_upper            = db.Column(db.Float)

    def to_dict(self):
        return {
            "id":                   self.id,
            "created_at":           self.created_at.isoformat(),
            "team_size":            self.team_size,
            "duration_months":      self.duration_months,
            "budget_usd":           self.budget_usd,
            "requirements_clarity": self.requirements_clarity,
            "client_involvement":   self.client_involvement,
            "tech_complexity":      self.tech_complexity,
            "risk_level":           self.risk_level,
            "project_type":         self.project_type,
            "recommended_sdlc":     self.recommended_sdlc,
            "confidence":           self.confidence,
            "alternatives":         self.alternatives.split(",") if self.alternatives else [],
            "effort_person_months": self.effort_person_months,
            "estimated_cost_usd":   self.estimated_cost_usd,
            "cost_lower":           self.cost_lower,
            "cost_upper":           self.cost_upper,
        }
