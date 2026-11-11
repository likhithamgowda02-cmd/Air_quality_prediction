
import streamlit as st
import pandas as pd
import joblib

# Load the trained machine learning model
model = joblib.load("air_quality_model_compressed.pkl")
features = joblib.load("air_quality_features.pkl")

# Configure the application
st.set_page_config(
    page_title="Air Quality Analytics",
    page_icon="🌍",
    layout="wide"
)

# Sidebar navigation
st.sidebar.title("🌍 Air Quality System")
st.sidebar.write("Machine Learning Dashboard")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Prediction",
        "Analytics",
        "Model Performance",
        "Feature Importance",
    ]
)

# Home page
if page == "Home":

    st.title("🌍 Air Quality Analytics Dashboard")

    st.write(
        "Machine Learning Based CO Concentration Prediction"
    )

    st.divider()

    st.subheader("Project Overview")

    st.write("""
    This application uses a Random Forest Regression model
    to estimate carbon monoxide concentration from environmental
    and gas sensor measurements.
    """)

    col1, col2, col3 = st.columns(3)

    col1.metric("ML Algorithm", "Random Forest")
    col2.metric("Input Features", len(features))
    col3.metric("Prediction Target", "CO(GT)")

    st.divider()

    st.subheader("Available Modules")

    st.write("""
    - Home: Project overview
    - Prediction: Estimate CO concentration
    - Analytics: Explore environmental readings
    - Model Performance: Evaluate machine learning results
    - Feature Importance: Understand model inputs
    - Prediction History: Track previous predictions
    """)

# Other pages - temporary placeholders

elif page == "Prediction":
    st.header("CO Concentration Prediction")
    st.write("Enter environmental and gas sensor readings to predict CO concentration.")

    with st.form("prediction_form"):

        col1, col2 = st.columns(2)

        with col1:
            s1 = st.number_input("PT08.S1 (CO)", value=1000.0)
            s2 = st.number_input("PT08.S2 (NMHC)", value=1000.0)
            s3 = st.number_input("PT08.S3 (NOx)", value=1000.0)
            s4 = st.number_input("PT08.S4 (NO2)", value=1000.0)

        with col2:
            s5 = st.number_input("PT08.S5 (O3)", value=1000.0)
            temp = st.number_input("Temperature (°C)", value=20.0)
            rh = st.number_input("Relative Humidity (%)", value=50.0)
            ah = st.number_input("Absolute Humidity", value=1.0)

        submit = st.form_submit_button("Predict CO Concentration")

    if submit:

        input_data = pd.DataFrame([[
            s1, s2, s3, s4, s5, temp, rh, ah
        ]], columns=features)

        prediction = model.predict(input_data)[0]

        st.success("Prediction completed successfully!")

        st.metric(
            label="Predicted CO Concentration",
            value=f"{prediction:.3f} mg/m³"
        )



elif page == "Analytics":
    st.title("Air Quality Analytics")
    st.write("Explore environmental sensor readings and CO concentration.")

    st.subheader("CO Concentration Analysis")

    st.info(
        "Dataset required: Upload the original AirQualityUCI.xlsx "
        "file to view actual dataset analytics."
    )

    uploaded_file = st.file_uploader(
        "Upload Air Quality Dataset",
        type=["xlsx", "csv"]
    )

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".xlsx"):
                df = pd.read_excel(uploaded_file, na_values=["-200"])
            else:
                df = pd.read_csv(uploaded_file, na_values=["-200"])

            df.columns = df.columns.str.strip()

            if "CO(GT)" not in df.columns:
                st.error("CO(GT) column not found. Please upload the UCI dataset.")
            else:
                df["CO(GT)"] = pd.to_numeric(
                    df["CO(GT)"], errors="coerce"
                )
                df = df.dropna(subset=["CO(GT)"])
                df = df[df["CO(GT)"] >= 0]

                st.success("Dataset loaded successfully!")

                col1, col2, col3 = st.columns(3)

                col1.metric("Total Records", len(df))
                col2.metric(
                    "Average CO",
                    f"{df['CO(GT)'].mean():.2f}"
                )
                col3.metric(
                    "Maximum CO",
                    f"{df['CO(GT)'].max():.2f}"
                )

                st.subheader("CO Concentration Distribution")
                st.bar_chart(df["CO(GT)"].reset_index(drop=True))

                st.subheader("Temperature Analysis")
                if "T" in df.columns:
                    df["T"] = pd.to_numeric(df["T"], errors="coerce")
                    st.line_chart(df["T"].dropna().reset_index(drop=True))

                st.subheader("Humidity Analysis")
                if "RH" in df.columns:
                    df["RH"] = pd.to_numeric(df["RH"], errors="coerce")
                    st.line_chart(df["RH"].dropna().reset_index(drop=True))

                st.subheader("Sensor Correlation")
                numeric_df = df.select_dtypes(include="number")
                st.dataframe(
                    numeric_df.corr().round(2),
                    use_container_width=True
                )

        except Exception as e:
            st.error(f"Unable to process dataset: {e}")



elif page == "Model Performance":
    st.title("Model Performance Evaluation")
    st.write("Evaluation of machine learning algorithms.")

    st.subheader("Random Forest Regression")

    col1, col2, col3 = st.columns(3)

    col1.metric("MAE", "0.4342")
    col2.metric("RMSE", "0.6656")
    col3.metric("R² Score", "0.7515")

    st.success("Random Forest Regression achieved the best performance.")

    st.subheader("Algorithm Comparison")

    performance = pd.DataFrame({
        "Algorithm": [
            "Random Forest",
            "Decision Tree",
            "Linear Regression"
        ],
        "MAE": [0.4342, 0.5252, 0.4721],
        "RMSE": [0.6656, 0.8512, 0.7259],
        "R2 Score": [0.7515, 0.5936, 0.7045]
    })

    st.dataframe(performance, use_container_width=True)

    st.subheader("Model Accuracy Comparison")

    st.bar_chart(
        performance.set_index("Algorithm")[["R2 Score"]]
    )

    st.subheader("Error Comparison")

    st.bar_chart(
        performance.set_index("Algorithm")[["MAE", "RMSE"]]
    )

    st.info(
        "Higher R² and lower MAE/RMSE indicate better model performance."
    )


elif page == "Feature Importance":
    st.title("Feature Importance Analysis")

    st.write(
        "Understand which environmental parameters "
        "influence CO concentration predictions."
    )

    if hasattr(model, "feature_importances_"):

        importance = pd.DataFrame({
            "Feature": features,
            "Importance": model.feature_importances_
        })

        importance = importance.sort_values(
            by="Importance",
            ascending=False
        )

        st.subheader("Feature Importance Ranking")

        st.dataframe(importance, use_container_width=True)

        st.subheader("Feature Importance Chart")

        st.bar_chart(
            importance.set_index("Feature")
        )

        st.success(
            "Feature importance analysis completed."
        )

    else:
        st.warning(
            "The loaded model does not expose feature_importances_. "
            "Check whether it is a Random Forest model."
        )

