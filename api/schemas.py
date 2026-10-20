"""Request and response models for the prediction API.

Fields use readable names; `to_features` maps them to the UCI column names the model was
trained on. Bounds reject impossible values with HTTP 422 before the model is called.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# readable field name -> UCI column name
FIELD_TO_COLUMN = {
    "age": "age",
    "sex": "sex",
    "chest_pain_type": "cp",
    "resting_blood_pressure": "trestbps",
    "cholesterol": "chol",
    "fasting_blood_sugar_gt_120": "fbs",
    "resting_ecg": "restecg",
    "max_heart_rate": "thalach",
    "exercise_induced_angina": "exang",
    "st_depression": "oldpeak",
    "st_slope": "slope",
    "major_vessels": "ca",
    "thalassemia": "thal",
}


class PatientFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def reject_booleans(cls, data):
        # No field is boolean; without this, JSON true/false would pass as 1/0 for the coded fields.
        if isinstance(data, dict):
            bad = [k for k, v in data.items() if isinstance(v, bool)]
            if bad:
                raise ValueError(f"numeric codes expected, got true/false for: {', '.join(bad)}")
        return data

    age: int = Field(..., ge=18, le=100, description="Age in years")
    sex: Literal[0, 1] = Field(..., description="1 = male, 0 = female")
    chest_pain_type: Literal[1, 2, 3, 4] = Field(
        ..., description="1 typical angina, 2 atypical angina, 3 non-anginal pain, 4 asymptomatic"
    )
    resting_blood_pressure: float = Field(..., ge=70, le=250, description="mm Hg")
    cholesterol: float = Field(..., ge=80, le=700, description="Serum cholesterol, mg/dl")
    fasting_blood_sugar_gt_120: Literal[0, 1] = Field(..., description="1 if > 120 mg/dl")
    resting_ecg: Literal[0, 1, 2] = Field(
        ..., description="0 normal, 1 ST-T abnormality, 2 left ventricular hypertrophy"
    )
    max_heart_rate: float = Field(..., ge=50, le=250, description="Maximum heart rate achieved")
    exercise_induced_angina: Literal[0, 1] = Field(..., description="1 = yes")
    st_depression: float = Field(..., ge=0, le=10, description="ST depression by exercise")
    st_slope: Literal[1, 2, 3] = Field(..., description="1 upsloping, 2 flat, 3 downsloping")
    major_vessels: int | None = Field(
        None, ge=0, le=3, description="Vessels coloured by fluoroscopy; null if unknown"
    )
    thalassemia: Literal[3, 6, 7] | None = Field(
        None, description="3 normal, 6 fixed defect, 7 reversible defect; null if unknown"
    )

    def to_features(self) -> dict:
        return {FIELD_TO_COLUMN[k]: v for k, v in self.model_dump().items()}


class Prediction(BaseModel):
    prediction: Literal[0, 1] = Field(..., description="1 = heart disease predicted")
    label: Literal["disease", "no disease"]
    probability_of_disease: float = Field(..., ge=0, le=1)
    confidence: float = Field(..., ge=0.5, le=1, description="Probability of the predicted class")
    model_version: str
