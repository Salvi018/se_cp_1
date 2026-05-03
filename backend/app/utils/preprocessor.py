import numpy as np

RISK_MAP = {"low": 1, "medium": 2, "high": 3}
TYPE_MAP = {"web": 1, "mobile": 2, "embedded": 3, "data": 4}
GEO_MAP  = {"co-located": 1, "hybrid": 2, "distributed": 3}

_INT_FIELDS = {
    "team_size":               (int, 1,  100),
    "duration_months":         (int, 1,   60),
    "requirements_clarity":    (int, 1,    5),
    "client_involvement":      (int, 1,    5),
    "tech_complexity":         (int, 1,    5),
    # new
    "team_experience":         (int, 1,    5),   # 1=junior, 5=expert
    "regulatory_compliance":   (int, 0,    1),   # 0=no, 1=yes (HIPAA/ISO/etc)
    "geographic_distribution": (int, 1,    3),   # 1=co-located,2=hybrid,3=distributed
}
_FLOAT_FIELDS = {
    "budget_usd": (float, 1, 8_350_000_000),  # accepts INR value directly
}
_CHOICE_FIELDS = {
    "risk_level":   list(RISK_MAP.keys()),
    "project_type": list(TYPE_MAP.keys()),
}


def validate_and_parse(data: dict) -> tuple[dict, list[str]]:
    errors = []
    parsed = {}

    for field, (ftype, lo, hi) in _INT_FIELDS.items():
        # New fields are optional — default values for backward compatibility
        if field not in data:
            if field == "team_experience":
                parsed[field] = 3        # default: mid-level
                continue
            if field == "regulatory_compliance":
                parsed[field] = 0        # default: no compliance
                continue
            if field == "geographic_distribution":
                parsed[field] = 1        # default: co-located
                continue
            errors.append(f"'{field}' is required")
            continue
        try:
            val = ftype(data[field])
        except (TypeError, ValueError):
            errors.append(f"'{field}' must be an integer")
            continue
        if not (lo <= val <= hi):
            errors.append(f"'{field}' must be between {lo} and {hi}, got {val}")
            continue
        parsed[field] = val

    for field, (ftype, lo, hi) in _FLOAT_FIELDS.items():
        if field not in data:
            errors.append(f"'{field}' is required")
            continue
        try:
            val = ftype(data[field])
        except (TypeError, ValueError):
            errors.append(f"'{field}' must be a number")
            continue
        if not (lo <= val <= hi):
            errors.append(f"'{field}' must be between {lo:,} and {hi:,}, got {val:,}")
            continue
        parsed[field] = val

    for field, choices in _CHOICE_FIELDS.items():
        if field not in data:
            errors.append(f"'{field}' is required")
            continue
        val = str(data[field]).lower()
        if val not in choices:
            errors.append(f"'{field}' must be one of {choices}, got '{val}'")
            continue
        parsed[field] = val

    return parsed, errors


USD_TO_INR = 83.5

def build_feature_vector(data: dict) -> np.ndarray:
    team_size            = float(data["team_size"])
    duration_months      = float(data["duration_months"])
    budget_usd           = float(data["budget_usd"]) / USD_TO_INR  # convert INR → USD
    requirements_clarity = float(data["requirements_clarity"])
    client_involvement   = float(data["client_involvement"])
    tech_complexity      = float(data["tech_complexity"])
    risk_encoded         = float(RISK_MAP[data["risk_level"]])
    type_encoded         = float(TYPE_MAP[data["project_type"]])
    budget_per_person    = budget_usd / max(team_size, 1)
    complexity_risk      = tech_complexity * risk_encoded
    # new features
    team_experience         = float(data.get("team_experience", 3))
    regulatory_compliance   = float(data.get("regulatory_compliance", 0))
    geographic_distribution = float(data.get("geographic_distribution", 1))

    return np.array([[
        team_size, duration_months, budget_usd,
        requirements_clarity, client_involvement, tech_complexity,
        risk_encoded, type_encoded, budget_per_person, complexity_risk,
        team_experience, regulatory_compliance, geographic_distribution,
    ]])
