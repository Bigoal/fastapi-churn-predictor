from fastapi import FastAPI
import joblib
import pandas as pd

# create app
app = FastAPI()
gender_map = {
    "Male": 1,
    "Female": 0
}

partner_map = {
    "Yes": 1,
    "No": 0
}

dependents_map = {
    "Yes": 1,
    "No": 0
}

binary_map = {
    "Yes": 1,
    "No": 0
}
# load model
data = joblib.load("churn_model.pkl")
model = data["model"]
features = data["features"]

@app.post("/predict")
def predict(input_data: dict):

    try:
        # 🔄 Convert values
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