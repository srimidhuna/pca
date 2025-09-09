## 🧪 Wine Quality Prediction App

This Gradio-based web application predicts wine quality (score 3–9) using physicochemical properties of wine samples. It also displays the probability distribution across quality classes, if supported by the model.

---

### 🚀 Features

- Predict wine quality (integer values from 3 to 9)
- Display class probabilities for each quality score
- PCA-based dimensionality reduction
- Scaled input features for consistent model performance
- Interactive Gradio interface

---

### 🛠️ Requirements

- Python 3.8+
- Required packages:
  ```bash
  pip install pandas joblib gradio scikit-learn
  ```

---

### 📁 Files

| File                          | Description                                                                 |
|-------------------------------|-----------------------------------------------------------------------------|
| `app.py`                      | Main Gradio application that launches the wine quality prediction UI       |
| `best_model.pkl`              | Trained machine learning model used for prediction                         |
| `preprocessing_artifacts.pkl` | Contains fitted scaler, PCA transformer, and feature column names          |
| `code.ipynb`                  | Jupyter notebook with full data preprocessing, PCA, model training, and evaluation |
| `winequality.csv`             | Dataset containing physicochemical properties and wine quality labels      |
| `README.md`                   | Project documentation and usage instructions                               |
| `requirements.txt`            | List of required Python packages for easy installation                  |

---

### 📊 Input Features

- Fixed Acidity  
- Volatile Acidity  
- Citric Acid  
- Residual Sugar  
- Chlorides  
- Free Sulfur Dioxide  
- Total Sulfur Dioxide  
- pH  
- Sulphates  
- Alcohol  

---

### 🎯 Output

- **Predicted Wine Quality** – Integer score from 3 to 9  
- **Class Probabilities** – Probability for each quality class (if supported)

---

### 🧬 Model Development Workflow (from `code.ipynb`)


Typical steps might include:
- Data loading and exploration
- Handling missing values
- Feature scaling using `StandardScaler`
- Dimensionality reduction using `PCA`
- Model training (e.g., RandomForestClassifier, GradientBoosting)
- Hyperparameter tuning
- Model evaluation (accuracy, confusion matrix)
- Saving model and preprocessing artifacts with `joblib`

---

### ▶️ How to Run

1. Ensure all files (`app.py`, `best_model.pkl`, `preprocessing_artifacts.pkl`) are in the same directory.
2. Launch the app:
   ```bash
   python app.py
   ```
3. Open the local Gradio URL in your browser.


APP LINK : https://huggingface.co/spaces/srimidhuna/wine_quality_prediction


Here is the interface of the Wine Quality Prediction App built using Gradio:

<img width="826" height="880" alt="Screenshot 2025-09-09 120316" src="https://github.com/user-attachments/assets/77985931-b647-48c1-a103-db915f33fda8" />


Wine Quality Prediction (Sample Output)

<img width="879" height="617" alt="Screenshot 2025-09-09 120548" src="https://github.com/user-attachments/assets/846b2865-c193-4964-acad-57d96bc33523" />

