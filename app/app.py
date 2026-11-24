from pathlib import Path

import joblib
import streamlit as st

st.set_page_config(page_title="Amazon Review Sentiment", layout="wide")

# Define the models and their configurations
MODELS = [
    {"label": "Naive Bayes (BoW)", "key": "nb_bow", "summary_repeats": 2},
    {"label": "Naive Bayes (TF-IDF)", "key": "nb_tfidf", "summary_repeats": 2},
    {"label": "Logistic Regression (BoW)", "key": "lr_bow", "summary_repeats": 1},
    {"label": "Logistic Regression (TF-IDF)", "key": "lr_tfidf", "summary_repeats": 1},
    {"label": "Feedforward NN (TF-IDF)", "key": "fnn_tfidf", "summary_repeats": 2},
]

# Define the keys for the models that will be used for predictions
PREDICTION_MODEL_KEYS = ("fnn_tfidf", "lr_tfidf")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@st.cache_resource
def load_prediction_models():
    return {
        model_key: (
            joblib.load(PROJECT_ROOT / "models" / f"{model_key}_model.joblib"),
            joblib.load(
                PROJECT_ROOT / "models" / f"{model_key}_vectorizer.joblib"
            ),
        )
        for model_key in PREDICTION_MODEL_KEYS
    }

# Streamlit app layout
st.title("Amazon Food Review Sentiment")
st.write(
    "Enter a review to compare sentiment predictions from the Feedforward Neural "
    "Network and TF-IDF Logistic Regression models."
)

with st.form("review_form"):
    review_text = st.text_area("Review text", height=150, key="review_text")
    review_summary = st.text_input("Review summary (optional)", key="review_summary")
    analyze = st.form_submit_button("Analyze sentiment", type="primary")

predict_tab, comparison_tab, about_tab = st.tabs(
    ["Predictions", "Model comparison", "About"]
)

# Sentiment analysis and display predictions
with predict_tab:
    if analyze:
        if not review_text.strip():
            st.warning("Please enter review text before analyzing.")
        else:
            with st.spinner("Loading models and analyzing the review..."):
                loaded_models = load_prediction_models()
                predictions = []
                model_by_key = {model["key"]: model for model in MODELS}
                for model_key in PREDICTION_MODEL_KEYS:
                    model, vectorizer = loaded_models[model_key]
                    config = model_by_key[model_key]
                    combined_text = (
                        f"{(review_summary or '').strip()} "
                        * config["summary_repeats"]
                        + review_text.strip()
                    ).strip()
                    features = vectorizer.transform([combined_text])
                    predicted_label = int(model.predict(features)[0])
                    positive_class_index = list(model.classes_).index(1)
                    positive_probability = float(
                        model.predict_proba(features)[0][positive_class_index]
                    )
                    predictions.append(
                        (
                            config["label"],
                            predicted_label,
                            positive_probability,
                        )
                    )

            result_columns = st.columns(len(predictions))
            for column, (label, predicted_label, positive_probability) in zip(
                result_columns, predictions
            ):
                with column:
                    st.subheader(label)
                    st.metric("Positive probability", f"{positive_probability:.1%}")
                    if predicted_label == 1:
                        st.success("Positive sentiment")
                    else:
                        st.error("Negative sentiment")
                    confidence = (
                        positive_probability
                        if predicted_label == 1
                        else 1 - positive_probability
                    )
                    st.caption(f"Prediction confidence: {confidence:.1%}")
    else:
        st.info("Enter a review above and select **Analyze sentiment** to get started.")

# Model comparison and information 
with comparison_tab:
    st.subheader("Available models")
    st.table(
        [
            {
                "Model": model["label"],
                "Key": model["key"],
                "Summary repeats": model["summary_repeats"],
            }
            for model in MODELS
        ]
    )
    st.caption(
        "Predictions are currently provided by Feedforward NN (TF-IDF) and "
        "Logistic Regression (TF-IDF)."
    )

# About tab with information on how the models work
with about_tab:
    st.subheader("How it works")
    st.write(
        "Each model uses its saved TF-IDF vectorizer to transform the review, "
        "then predicts negative (0) or positive (1) sentiment. If provided, the "
        "review summary is combined with the review using the same repetition "
        "count used during training."
    )
