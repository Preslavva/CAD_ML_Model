from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "cad_rf_pipeline.pkl")

# Loading the saved Random Forest model
model = joblib.load(MODEL_PATH)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction_text = None
    probability = None

    if request.method == "POST":
        trestbps = float(request.form["trestbps"])
        oldpeak = float(request.form["oldpeak"])
        sex = int(request.form["sex"])
        fbs = int(request.form["fbs"])
        slope = int(request.form["slope"])
        ca = int(request.form["ca"])
        restecg = int(request.form["restecg"])

        input_df = pd.DataFrame([{
            "trestbps": trestbps,
            "oldpeak": oldpeak,
            "sex": sex,
            "fbs": fbs,
            "slope": slope,
            "ca": ca,
            "restecg": restecg
        }])

        y_pred = model.predict(input_df)[0]
        y_proba = model.predict_proba(input_df)[0][1]

        if y_pred == 1:
            prediction_text = "Model prediction: Coronary Artery Disease (class 1)"
        else:
            prediction_text = "Model prediction: No Coronary Artery Disease (class 0)"

        probability = round(y_proba * 100, 2)

    return render_template(
        "index.html",
        prediction=prediction_text,
        probability=probability
    )

if __name__ == "__main__":
    app.run(debug=True)