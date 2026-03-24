# from fastapi import FastAPI
# import joblib
# import pandas as pd

# # create app
# app = FastAPI()

# # mappings
# gender_map = {"Male": 1, "Female": 0}
# partner_map = {"Yes": 1, "No": 0}
# dependents_map = {"Yes": 1, "No": 0}
# binary_map = {"Yes": 1, "No": 0}

# # load model
# data = joblib.load("churn_model.pkl")
# model = data["model"]
# features = data["features"]

# # POST route (original)
# @app.post("/predict")
# def predict(input_data: dict):
#     return make_prediction(input_data)

# # GET route (browser-friendly)
# @app.get("/predict")
# def predict_get(
#     gender: str,
#     Partner: str,
#     Dependents: str,
#     PhoneService: str,
#     PaperlessBilling: str,
#     TotalCharges: float
# ):
#     input_data = {
#         "gender": gender,
#         "Partner": Partner,
#         "Dependents": Dependents,
#         "PhoneService": PhoneService,
#         "PaperlessBilling": PaperlessBilling,
#         "TotalCharges": TotalCharges
#     }
#     return make_prediction(input_data)

# # shared prediction logic
# def make_prediction(input_data):
#     try:
#         # convert categorical values
#         input_data["gender"] = gender_map.get(input_data["gender"], input_data["gender"])
#         input_data["Partner"] = partner_map.get(input_data["Partner"], input_data["Partner"])
#         input_data["Dependents"] = dependents_map.get(input_data["Dependents"], input_data["Dependents"])
#         input_data["PhoneService"] = binary_map.get(input_data["PhoneService"], input_data["PhoneService"])
#         input_data["PaperlessBilling"] = binary_map.get(input_data["PaperlessBilling"], input_data["PaperlessBilling"])

#         df = pd.DataFrame([input_data])

#         for col in features:
#             if col not in df:
#                 df[col] = None

#         df = df[features]

#         df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

#         prediction = model.predict(df)[0]
#         probability = model.predict_proba(df)[0][1]

#         return {
#             "prediction": int(prediction),
#             "churn_probability": round(probability, 3)
#         }

#     except Exception as e:
#         return {"error": str(e)}
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import joblib
import pandas as pd

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

# ----- Shared prediction logic -----
def make_prediction(input_data):
    try:
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

# ----- Home page -----
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <head><title>Telco Churn Predictor</title></head>
        <body style="text-align:center; font-family:Arial">
            <h1>Welcome to the Telco Churn Predictor</h1>
            <img src="https://via.placeholder.com/400x200.png?text=Telco+Churn+Model" alt="Telco" />
            <br><br>
            <a href="/predict_page"><button style="padding:10px 20px; font-size:16px;">Go to Predict</button></a>
        </body>
    </html>
    """

# ----- Predict page with example buttons -----
@app.get("/predict_page", response_class=HTMLResponse)
def predict_page():
    return """
    <html>
        <head><title>Predict Churn</title></head>
        <body style="text-align:center; font-family:Arial">
            <h2>Select an Example</h2>
            <a href="/predict?gender=Male&Partner=Yes&Dependents=No&PhoneService=Yes&PaperlessBilling=No&TotalCharges=500">
                <button style="padding:10px 20px; margin:10px;">Example 1</button>
            </a>
            <a href="/predict?gender=Female&Partner=No&Dependents=Yes&PhoneService=No&PaperlessBilling=Yes&TotalCharges=200">
                <button style="padding:10px 20px; margin:10px;">Example 2</button>
            </a>
        </body>
    </html>
    """

# ----- GET predict route -----
@app.get("/predict")
def predict_get(
    gender: str,
    Partner: str,
    Dependents: str,
    PhoneService: str,
    PaperlessBilling: str,
    TotalCharges: float
):
    result = make_prediction({
        "gender": gender,
        "Partner": Partner,
        "Dependents": Dependents,
        "PhoneService": PhoneService,
        "PaperlessBilling": PaperlessBilling,
        "TotalCharges": TotalCharges
    })
    return result

# ----- POST predict route (API) -----
@app.post("/predict")
def predict_post(input_data: dict):
    return make_prediction(input_data)