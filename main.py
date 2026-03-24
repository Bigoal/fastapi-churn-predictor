from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import joblib
import pandas as pd

app = FastAPI()

# serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# load model
data = joblib.load("churn_model.pkl")
model = data["model"]
features = data["features"]

# all input features used by the Telco dataset
FEATURE_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]

binary_map = {"Yes": 1, "No": 0}
gender_map = {"Male": 1, "Female": 0}

def normalize_input(input_data: dict) -> dict:
    row = input_data.copy()

    # remove target if it is accidentally sent
    row.pop("Churn", None)

    # type conversions
    if "SeniorCitizen" in row:
        row["SeniorCitizen"] = int(row["SeniorCitizen"])

    if "tenure" in row:
        row["tenure"] = int(row["tenure"])

    if "MonthlyCharges" in row:
        row["MonthlyCharges"] = float(row["MonthlyCharges"])

    if "TotalCharges" in row:
        row["TotalCharges"] = float(row["TotalCharges"])

    # binary mappings
    if "gender" in row:
        row["gender"] = gender_map.get(row["gender"], row["gender"])

    for col in ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]:
        if col in row:
            row[col] = binary_map.get(row[col], row[col])

    return row

def make_prediction(input_data: dict):
    try:
        missing = [col for col in FEATURE_COLUMNS if col not in input_data]
        if missing:
            return {
                "error": f"Missing fields: {', '.join(missing)}"
            }

        row = normalize_input(input_data)

        df = pd.DataFrame([row])

        # keep only the columns the model expects, in the correct order
        df = df.reindex(columns=features)

        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]

        return {
            "prediction": int(prediction),
            "churn_probability": round(float(probability), 3)
        }

    except Exception as e:
        return {"error": str(e)}

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <head>
            <title>Telco Churn Predictor</title>
        </head>
        <body style="text-align:center; font-family:Arial; padding:40px;">
            <h1>Welcome to the Telco Churn Predictor</h1>
            <img src="/static/photo.jpg" alt="Photo" style="width:300px; border-radius:12px; margin:20px 0;" />
            <br>
            <a href="/predict_page">
                <button style="padding:12px 24px; font-size:16px; cursor:pointer;">Go to Predict</button>
            </a>
            <footer style="margin-top:50px; color:gray;">
                <p><b>Created by Bigoal Shereen</b></p>
            </footer>
        </body>
    </html>
    """

@app.get("/predict_page", response_class=HTMLResponse)
def predict_page():
    return """
    <html>
        <head>
            <title>Predict Churn</title>
        </head>
        <body style="text-align:center; font-family:Arial; padding:40px;">
            <h2>Select an Example</h2>

            <a href="/predict?gender=Male&SeniorCitizen=0&Partner=No&Dependents=No&tenure=1&PhoneService=No&MultipleLines=No%20phone%20service&InternetService=Fiber%20optic&OnlineSecurity=No&OnlineBackup=No&DeviceProtection=No&TechSupport=No&StreamingTV=Yes&StreamingMovies=Yes&Contract=Month-to-month&PaperlessBilling=Yes&PaymentMethod=Electronic%20check&MonthlyCharges=79.85&TotalCharges=79.85">
                <button style="padding:10px 20px; margin:10px; cursor:pointer;">Example 1</button>
            </a>

            <a href="/predict?gender=Female&SeniorCitizen=0&Partner=Yes&Dependents=Yes&tenure=60&PhoneService=Yes&MultipleLines=Yes&InternetService=DSL&OnlineSecurity=Yes&OnlineBackup=Yes&DeviceProtection=Yes&TechSupport=Yes&StreamingTV=Yes&StreamingMovies=Yes&Contract=Two%20year&PaperlessBilling=No&PaymentMethod=Credit%20card%20%28automatic%29&MonthlyCharges=59.85&TotalCharges=3500.0">
                <button style="padding:10px 20px; margin:10px; cursor:pointer;">Example 2</button>
            </a>

            <footer style="margin-top:50px; color:gray;">
                <p><b>Created by Bigoal Shereen</b></p>
            </footer>
        </body>
    </html>
    """

@app.get("/predict")
def predict_get(
    gender: str,
    SeniorCitizen: int,
    Partner: str,
    Dependents: str,
    tenure: int,
    PhoneService: str,
    MultipleLines: str,
    InternetService: str,
    OnlineSecurity: str,
    OnlineBackup: str,
    DeviceProtection: str,
    TechSupport: str,
    StreamingTV: str,
    StreamingMovies: str,
    Contract: str,
    PaperlessBilling: str,
    PaymentMethod: str,
    MonthlyCharges: float,
    TotalCharges: float
):
    input_data = {
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": tenure,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges
    }
    return make_prediction(input_data)

@app.post("/predict")
def predict_post(input_data: dict):
    return make_prediction(input_data)