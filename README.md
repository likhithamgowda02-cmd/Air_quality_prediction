# Air Quality Prediction System

## Project Overview
The Air Quality Prediction System is a Machine Learning-based application that predicts carbon monoxide (CO) concentration using environmental sensor parameters and historical air quality data.

The project uses a Random Forest Regression algorithm to estimate CO concentration and provides an interactive dashboard developed using Streamlit.

## Objectives
- Predict CO concentration using Machine Learning.
- Analyze environmental air quality parameters.
- Provide an interactive user-friendly dashboard.
- Visualize prediction results and model performance.

## Technologies Used
- Python
- Machine Learning
- Random Forest Regression
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib
- Joblib

## Dataset
UCI Air Quality Dataset.

## Model Performance
- MAE: 0.4342
- RMSE: 0.6656
- R² Score: 0.7515

## Project Structure
- app.py – Streamlit application
- air_quality_model_compressed.pkl – Trained ML model
- air_quality_features.pkl – Model feature information
- requirements.txt – Required Python libraries
- README.md – Project documentation

## Installation and Execution

1. Clone the repository:
   `git clone YOUR_GITHUB_REPOSITORY_URL`

2. Install dependencies:
   `pip install -r requirements.txt`

3. Run the application:
   `streamlit run app.py`

## Application
This application provides estimated CO concentration predictions. It is an academic prediction tool and not an official air quality monitoring or AQI measurement system.

## Developed By
Likhitha H M

Electronics and Communication Engineering
