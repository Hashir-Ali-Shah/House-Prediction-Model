# California House Price Prediction

This project is a machine learning application designed to predict median house prices in California. It provides an interactive interface built with Streamlit, allowing users to explore different algorithms, tune hyperparameters, and visualize model performance.

The project is structured modularly to ensure clean separation between data processing, model implementation, and the user interface.

## Table of Contents
1. [Features](#features)
2. [Project Structure](#project-structure)
3. [Code Documentation](#code-documentation)
4. [How to Run](#how-to-run)
5. [Model Interpretation](#model-interpretation)

## Features
- **Interactive UI**: Select and train algorithms (Linear Regression, Decision Trees, Random Forest) via a Streamlit sidebar.
- **Modular Design**: Separated concerns for data engineering and machine learning logic.
- **Data Persistence**: Preprocessed data and trained models are cached using `joblib` to speed up subsequent runs.
- **Advanced Visualizations**:
    - Interactive Actual vs. Predicted line charts.
    - Scatter plots with "Perfect Prediction" reference lines.
    - Cross-validation support for realistic Decision Tree performance visualization.

## Project Structure
```
d:/ML/Tasks/HousePrediction/
├── app.py                  # Streamlit application entry point
├── src/
│   ├── data.py             # Data loading and preprocessing pipeline
│   └── models.py           # Model training and evaluation logic
├── model_store/            # Directory for saved models and preprocessed data
├── Data/
│   └── housing.csv         # Raw dataset
└── README.md               # Project documentation
```

## Code Documentation

### 1. `app.py`
This is the main entry point of the application. It orchestrates the following flow:
- **Configuration**: Uses Streamlit widgets in the sidebar to capture user inputs (Algorithm choice, Random Forest estimators).
- **Data Orchestration**: Calls `src.data` to load and preprocess data. It checks the `model_store/` directory first to see if a cached version of the preprocessed data exists.
- **Training**: Initiates training via `src.models`. It includes a "Force Retrain" option to bypass the model cache.
- **Visualization**: Uses `matplotlib` and `streamlit` to generate performance metrics (RMSE) and diagnostic plots.

### 2. `src/data.py`
This module handles all data engineering tasks:
- **`load_data()`**: Loads the raw CSV into a Pandas DataFrame.
- **`CombinedAttributesAdder`**: A custom Scikit-Learn transformer that creates new features such as `rooms_per_household`, `population_per_household`, and `bedrooms_per_room`.
- **Pre-processing Pipeline**: Uses `ColumnTransformer` to apply:
    - **Numeric pipeline**: Median imputation and Standard Scaling.
    - **Categorical pipeline**: One-Hot Encoding for the `ocean_proximity` feature.
- **`preprocess_data()`**: Combines the above into a single function that returns feature matrices and labels.

### 3. `src/models.py`
This module encapsulates the machine learning logic:
- **Training Wrappers**: Standardized functions for training `LinearRegression`, `DecisionTreeRegressor`, and `RandomForestRegressor`.
- **Evaluation Utilities**:
    - **`evaluate_model()`**: Calculates basic RMSE on the training set.
    - **`cross_validate_model()`**: Provides robust performance scores using K-Fold cross-validation.
    - **`get_cv_predictions()`**: Generates "out-of-sample" predictions for plotting, which is crucial for identifying overfitting in models like Decision Trees.

### 4. Persistence with `joblib`
Both `app.py` and the data pipeline use `joblib` for serialization. This ensures that:
- Heavy preprocessing is done once.
- Trained models can be shared and reused instantly across sessions.

## How to Run
1. Ensure you have Python installed.
2. Install the required dependencies:
   ```bash
   pip install streamlit pandas numpy matplotlib scikit-learn joblib
   ```
3. Run the application:
   ```bash
   streamlit run app.py
   ```

## Model Interpretation
In the scatter plot visualization:
- **X-axis**: Actual house price.
- **Y-axis**: Predicted house price.
- **Red Dashed Line**: Represents perfect prediction.
- **Blue Dots**: Predictions. If points cluster tightly around the red line, the model is performing accurately.
