
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from src.data import load_data, preprocess_data
from src.models import train_linear_regression, train_decision_tree, train_random_forest, evaluate_model, cross_validate_model, get_cv_predictions

st.set_page_config(page_title="House Prediction App", layout="wide")

st.title("🏡 California House Price Prediction")
st.markdown("""
This app explores different machine learning algorithms for predicting house prices in California.
Data processing and model training are modularized in the `src` directory.
""")

# Sidebar
st.sidebar.header("Configuration")
algo_option = st.sidebar.selectbox(
    "Select Algorithm",
    ("Linear Regression", "Decision Tree Regressor", "Random Forest Regressor")
)

# Hyperparameters for Random Forest
n_estimators = 100
if algo_option == "Random Forest Regressor":
    n_estimators = st.sidebar.slider("Number of Estimators", 10, 200, 30, 10)

# Load Data
@st.cache_data
def get_data():
    df = load_data()
    return df

try:
    housing = get_data()
    
    # Preprocessing
    st.subheader("1. Data Exploration")
    if st.checkbox("Show Raw Data"):
        st.write(housing.head())
    
    st.write("Data Description:")
    st.write(housing.describe())

    # Persistence Paths
    MODEL_DIR = "model_store"
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        
    DATA_PATH = os.path.join(MODEL_DIR, "data.joblib")
    LABELS_PATH = os.path.join(MODEL_DIR, "labels.joblib")

    # Preprocess Logic
    if os.path.exists(DATA_PATH) and os.path.exists(LABELS_PATH):
        st.info("Loading preprocessed data from disk...")
        housing_prepared = joblib.load(DATA_PATH)
        housing_labels = joblib.load(LABELS_PATH)
        st.success("Data Loaded!")
    else:
        with st.spinner("Preprocessing data..."):
            housing_prepared, housing_labels, pipeline = preprocess_data(housing)
            joblib.dump(housing_prepared, DATA_PATH)
            joblib.dump(housing_labels, LABELS_PATH)
        st.success("Data Preprocessed and Saved!")
    
    # Model Training
    st.subheader(f"2. Training: {algo_option}")
    
    # Define model filename based on config
    model_filename = algo_option.replace(" ", "_")
    if algo_option == "Random Forest Regressor":
        model_filename += f"_n{n_estimators}"
    model_path = os.path.join(MODEL_DIR, f"{model_filename}.joblib")
    
    force_retrain = st.checkbox("Force Retrain")
    
    model = None
    if st.button("Run"):
        with st.spinner("Processing..."):
            
            if os.path.exists(model_path) and not force_retrain:
                st.info(f"Loading {algo_option} model from disk...")
                model = joblib.load(model_path)
                st.success("Model Loaded!")
            else:
                st.info(f"Training {algo_option}...")
                if algo_option == "Linear Regression":
                    model = train_linear_regression(housing_prepared, housing_labels)
                elif algo_option == "Decision Tree Regressor":
                    model = train_decision_tree(housing_prepared, housing_labels)
                elif algo_option == "Random Forest Regressor":
                    model = train_random_forest(housing_prepared, housing_labels, n_estimators=n_estimators)
                
                joblib.dump(model, model_path)
                st.success("Model Trained and Saved!")
            
            # Evaluation
            rmse = evaluate_model(model, housing_prepared, housing_labels)
            st.metric(label="Training RMSE (In-Sample)", value=f"${rmse:,.2f}")
            
            # Predictions vs Actual
            if algo_option == "Decision Tree Regressor":
                st.warning("Note: Decision Trees can easily overfit the training data (memorize it), resulting in a perfect line. Below, we use Cross-Validation to show more realistic 'out-of-sample' predictions.")
                
                # Check if CV predictions are cached
                cv_pred_path = os.path.join(MODEL_DIR, f"{model_filename}_cv_predictions.joblib")
                if os.path.exists(cv_pred_path) and not force_retrain:
                    st.info("Loading cached cross-validation predictions...")
                    predictions = joblib.load(cv_pred_path)
                else:
                    with st.spinner("Calculating Cross-Validation predictions..."):
                        predictions = get_cv_predictions(model, housing_prepared, housing_labels)
                        joblib.dump(predictions, cv_pred_path)
            else:
                predictions = model.predict(housing_prepared)
            
            st.subheader("3. Actual vs Predicted (First 100 samples)")
            chart_data = pd.DataFrame({
                "Actual": housing_labels[:100],
                "Predicted": predictions[:100]
            })
            st.line_chart(chart_data)
            
            # Scatter plot
            fig, ax = plt.subplots()
            ax.scatter(housing_labels[:500], predictions[:500], alpha=0.4, s=10, label="Predictions")
            ax.plot([housing_labels.min(), housing_labels.max()], [housing_labels.min(), housing_labels.max()], 'r--', linewidth=2, label="Perfect Prediction")
            ax.set_xlabel("Actual Price ($)")
            ax.set_ylabel("Predicted Price ($)")
            title_suffix = "(Cross-Validated)" if algo_option == "Decision Tree Regressor" else "(Training Data)"
            ax.set_title(f"Actual vs Predicted {title_suffix}")
            ax.legend()
            st.pyplot(fig)
            
            st.markdown("""
            **How to Read This Graph:**
            
            This scatter plot shows how well the model's predictions match reality.
            
            - **X-axis (Horizontal)**: The **actual price** the house sold for (ground truth)
            - **Y-axis (Vertical)**: The **predicted price** from our model
            - 🔵 **Blue Points**: Each dot represents one house from the dataset
            - 🔴 **Red Dashed Line**: The "perfect prediction" line (where predicted = actual)
            
            **What to look for:**
            - **Points close to the red line**: Good predictions! The model got it right.
            - **Points above the red line**: The model **overestimated** (predicted higher than actual)
            - **Points below the red line**: The model **underestimated** (predicted lower than actual)
            - **Tight clustering around the line**: The model is accurate and consistent
            - **Wide scatter**: The model has high error/variance
            
            The actual prices naturally vary (that's why we have different houses at different price points on the X-axis). 
            A good model will have predictions that cluster tightly around the red line.
            """)

except FileNotFoundError:
    st.error("Data file not found. Please ensure 'Data/housing.csv' exists.")
except Exception as e:
    st.error(f"An error occurred: {e}")
