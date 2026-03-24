from fastapi import FastAPI
import joblib
import pandas as pd

# create app
app = FastAPI()

# mappings
gender_map = {"Male": 1, "Female": 0}
partner_map = {"Yes": 1, "No": 0}
dependents_map = {"Yes": 1, "No": 0}
binary_map = {"Yes": 1, "No": 0}

# load model
data = joblib.load("churn_model.pkl")
model = data["model"]
features = data["features"]

# POST route (original)
@app.post("/predict")
def predict(input_data: dict):
    return make_prediction(input_data)

# GET route (browser-friendly)
@app.get("/predict")
def predict_get(
    gender: str,
    Partner: str,
    Dependents: str,
    PhoneService: str,
    PaperlessBilling: str,
    TotalCharges: float
):
    input_data = {
        "gender": gender,
        "Partner": Partner,
        "Dependents": Dependents,
        "PhoneService": PhoneService,
        "PaperlessBilling": PaperlessBilling,
        "TotalCharges": TotalCharges
    }
    return make_prediction(input_data)

# shared prediction logic
def make_prediction(input_data):
    try:
        # convert categorical values
        input_data["gender"] = gender_map.get(input_data["gender"], input_data["gender"])
        input_data["Partner"] = partner_map.get(input_data["Partner"], input_data["Partner"])
        input_data["Dependents"] = dependents_map.get(input_data["Dependents"], input_data["Dependents"])
        input_data["PhoneService"] = binary_map.get(input_data["PhoneService"], input_data["PhoneService"])
        input_data["PaperlessBilling"] = binary_map.get(input_data["PaperlessBilling"], input_data["PaperlessBilling"])

        df = pd.DataFrame([input_data])

        for col in features:
            if col not in df:
                df[col] = None

        df = df[features]

        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]

        return {
            "prediction": int(prediction),
            "churn_probability": round(probability, 3)
        }

    except Exception as e:
        return {"error": str(e)}