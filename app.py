import gradio as gr
import pandas as pd
import joblib

model = joblib.load("best_model.pkl")
artifacts = joblib.load("preprocessing_artifacts.pkl")
scaler = artifacts["scaler"]
pca = artifacts["pca"]
feature_columns = artifacts["feature_columns"]

def predict_quality(fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, ph, sulphates, alcohol):
    input_df = pd.DataFrame([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, ph, sulphates, alcohol]], columns=feature_columns)
    X_scaled = scaler.transform(input_df)
    X_pca = pca.transform(X_scaled)
    prediction = model.predict(X_pca)[0]
    
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_pca)[0]
        labels = model.classes_.tolist()
        return int(prediction), {str(labels[i]): round(float(proba[i]), 3) for i in range(len(labels))}
    else:
        return int(prediction), {}

inputs = [
    gr.Number(label="Fixed Acidity"),
    gr.Number(label="Volatile Acidity"),
    gr.Number(label="Citric Acid"),
    gr.Number(label="Residual Sugar"),
    gr.Number(label="Chlorides"),
    gr.Number(label="Free Sulfur Dioxide"),
    gr.Number(label="Total Sulfur Dioxide"),
    gr.Number(label="pH"),
    gr.Number(label="Sulphates"),
    gr.Number(label="Alcohol")
]

outputs = [
    gr.Number(label="Predicted Wine Quality"),
    gr.Label(label="Class Probabilities")
]

app = gr.Interface(
    fn=predict_quality,
    inputs=inputs,
    outputs=outputs,
    title="Wine Quality Prediction",
    description="Predict wine quality (3–9) with probabilities."
)

if __name__ == "__main__":
    app.launch()




