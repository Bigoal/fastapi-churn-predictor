# 🚀 Telco Churn Predictor

A FastAPI web application that predicts telecom customer churn using a trained machine learning model.  
The project includes a simple browser interface, two built-in customer examples, and a live deployment on Render.

## 🌐 Live Demo

**Deployed on Render:**  
https://fastapi-churn-predictor.onrender.com

## 📌 Project Overview

This project was built as part of a machine learning workflow focused on **customer churn prediction**.  
During the notebook phase, I trained and compared several models, with a primary focus on improving **Recall** while maintaining strong overall performance. The objective was to identify as many churn customers as possible, as missing potential churners is more costly than misclassifying non-churn customers.

## ✨ Features

- 🌟 FastAPI backend
- 🤖 Machine learning model loaded from `churn_model.pkl`
- 🧭 Browser-friendly `/predict` endpoint
- 🖱️ Two pre-built example buttons for quick testing
- 🖼️ Static image support from the `static/` folder
- ☁️ Deployed and accessible on Render

## 🛠️ Tech Stack

- Python
- FastAPI
- Render
- Pandas
- Joblib
- Scikit-learn
- lightgbm
- xgboost
- catboost

## 📂 Project Structure

```text
fastapi-churn-predictor/
├── main.py
├── Telco_Churn.ipynb
├── churn_model.pkl
├── requirements.txt
├── .gitignore
├── Dockerfile
├── static/
│   └── photo.jpg
└── README.md
```

## 🧠 How It Works

1. Open the home page.
2. Click **Go to Predict**.
3. Choose one of the example buttons.
4. The app sends the customer data to `/predict`.
5. The model returns:
   - `prediction`
   - `churn_probability`

## 🔗 Browser Usage

You can test the prediction endpoint directly from the browser using query parameters.

Example:

```text
/predict?gender=Male&SeniorCitizen=0&Partner=No&Dependents=No&tenure=1&PhoneService=No&MultipleLines=No%20phone%20service&InternetService=Fiber%20optic&OnlineSecurity=No&OnlineBackup=No&DeviceProtection=No&TechSupport=No&StreamingTV=Yes&StreamingMovies=Yes&Contract=Month-to-month&PaperlessBilling=Yes&PaymentMethod=Electronic%20check&MonthlyCharges=79.85&TotalCharges=79.85
```

## 📊 Example Response

```json
{
  "prediction": 1,
  "churn_probability": 0.842
}
```

## ☁️ Deployment

This project is deployed on **Render** as a web service.

Typical Render settings:

- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`


## 👤 Author

**Bigoal Shereen**

---

## ⭐ Acknowledgment

Dataset provided by Kaggle.

Dataset link: https://www.kaggle.com/datasets/blastchar/telco-customer-churn.
