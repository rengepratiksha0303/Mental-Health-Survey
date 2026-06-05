import streamlit as st
import pandas as pd
import pickle

# Load Model Files
model = pickle.load(open("knn_model.pkl", "rb"))
scaler = pickle.load(open("knn_scaler.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

st.set_page_config(
    page_title="Mental Health Risk Predictor",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Mental Health Risk Prediction")
st.write("Predict Mental Health Risk using Machine Learning (KNN Model)")

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    try:
        participant_ids = None

        if "participant_id" in df.columns:
            participant_ids = df["participant_id"]

        drop_cols = []

        if "participant_id" in df.columns:
            drop_cols.append("participant_id")

        if "survey_date" in df.columns:
            drop_cols.append("survey_date")

        df = df.drop(columns=drop_cols)

        df_encoded = pd.get_dummies(df, drop_first=True)

        st.subheader("Processed Data")
        st.dataframe(df_encoded.head())

        X_scaled = scaler.transform(df_encoded)

        prediction = model.predict(X_scaled)

        prediction_labels = label_encoder.inverse_transform(
            prediction
        )

        result = pd.DataFrame()

        if participant_ids is not None:
            result["participant_id"] = participant_ids

        result["Predicted_Risk"] = prediction_labels

        st.subheader("Prediction Result")
        st.dataframe(result)

        csv = result.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Prediction CSV",
            data=csv,
            file_name="mental_health_predictions.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Error: {e}")

else:
    st.info("Upload a CSV file for prediction.")
