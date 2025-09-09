import gradio as gr
import pandas as pd
import joblib

model = joblib.load("best_model.pkl")
artifacts = joblib.load("preprocessing_artifacts.pkl")
scaler = artifacts["scaler"]
pca = artifacts["pca"]
feature_columns = artifacts["feature_columns"]

quality_labels = {
    3: "Very Poor",
    4: "Poor",
    5: "Average",
    6: "Good",
    7: "Very Good",
    8: "Excellent",
    9: "Outstanding"
}

def predict_quality(fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, ph, sulphates, alcohol):
    input_df = pd.DataFrame([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, ph, sulphates, alcohol]], columns=feature_columns)
    X_scaled = scaler.transform(input_df)
    X_pca = pca.transform(X_scaled)
    prediction = int(model.predict(X_pca)[0])

    label_text = f"{quality_labels.get(prediction, 'Unknown')} ({prediction})"

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_pca)[0]
        labels = model.classes_.tolist()
        proba_dict = {f"{quality_labels.get(lbl, lbl)} ({lbl})": round(float(proba[i]), 3) for i, lbl in enumerate(labels)}
        return label_text, proba_dict
    else:
        return label_text, {}

inputs = [
    gr.Number(label="Fixed Acidity", minimum=3.8, maximum=14.2),
    gr.Number(label="Volatile Acidity", minimum=0.08, maximum=1.1),
    gr.Number(label="Citric Acid", minimum=0.0, maximum=1.66),
    gr.Number(label="Residual Sugar", minimum=0.6, maximum=65.8),
    gr.Number(label="Chlorides", minimum=0.009, maximum=0.346),
    gr.Number(label="Free Sulfur Dioxide", minimum=2.0, maximum=289.0),
    gr.Number(label="Total Sulfur Dioxide", minimum=9.0, maximum=440.0),
    gr.Number(label="pH", minimum=2.72, maximum=3.82),
    gr.Number(label="Sulphates", minimum=0.22, maximum=1.08),
    gr.Number(label="Alcohol", minimum=8.0, maximum=14.2)
]

outputs = [
    gr.Textbox(label="Predicted Wine Quality"),
    gr.Label(label="Class Probabilities")
]

app = gr.Interface(
    fn=predict_quality,
    inputs=inputs,
    outputs=outputs,
    title="Wine Quality Prediction",
    description="Predict wine quality (3–9) with descriptive labels and probabilities."
)

if __name__ == "__main__":
    app.launch()




