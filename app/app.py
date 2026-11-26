from pathlib import Path

import joblib
import streamlit as st

st.set_page_config(
    page_title="Amazon Review Sentiment",
    page_icon="🥫",
    layout="wide",
)

MODELS = [
    {
        "label": "Naive Bayes",
        "key": "nb_bow",
        "family": "Naive Bayes",
        "representation": "Bag of Words",
        "summary_repeats": 2,
    },
    {
        "label": "Naive Bayes",
        "key": "nb_tfidf",
        "family": "Naive Bayes",
        "representation": "TF-IDF",
        "summary_repeats": 2,
    },
    {
        "label": "Logistic Regression",
        "key": "lr_bow",
        "family": "Logistic Regression",
        "representation": "Bag of Words",
        "summary_repeats": 1,
    },
    {
        "label": "Logistic Regression",
        "key": "lr_tfidf",
        "family": "Logistic Regression",
        "representation": "TF-IDF",
        "summary_repeats": 1,
    },
    {
        "label": "Feedforward Neural Network",
        "key": "fnn_tfidf",
        "family": "Neural Network",
        "representation": "TF-IDF",
        "summary_repeats": 2,
    },
]
PREDICTION_MODEL_KEYS = (
    "nb_bow",
    "nb_tfidf",
    "lr_bow",
    "lr_tfidf",
    "fnn_tfidf",
)
SAMPLE_REVIEWS = [
    {
        "button": "Delicious cookies",
        "summary": "Delicious.",
        "text": "These cookies are delicious and soft, my whole family loves them. "
        "Will buy again!",
    },
    {
        "button": "Moldy product",
        "summary": "Awful.",
        "text": "Terrible product. The package arrived moldy and tasted rancid. "
        "I want a refund.",
    },
    {
        "button": "Crushed delivery",
        "summary": "",
        "text": "Tastes good but the box arrived crushed and the shipping took forever.",
    },
    {
        "button": "Bitter coffee",
        "summary": "Not good.",
        "text": "I wanted to like this coffee but it was not good at all, very bitter.",
    },
]
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@st.cache_resource
def load_prediction_models():
    return {
        model_key: (
            joblib.load(PROJECT_ROOT / "models" / f"{model_key}_model.joblib"),
            joblib.load(PROJECT_ROOT / "models" / f"{model_key}_vectorizer.joblib"),
        )
        for model_key in PREDICTION_MODEL_KEYS
    }


st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(ellipse at 12% 0%, rgba(34, 120, 103, 0.10), transparent 34%),
            #f6f8f7;
    }
    [data-testid="stHeader"] { background: transparent; }
    .block-container { max-width: 1120px; padding-top: 2.4rem; }
    h1 { letter-spacing: -0.035em; }
    div[data-testid="stMetric"] {
        background: #f4f8f6;
        border: 1px solid #e1ebe6;
        border-radius: 12px;
        padding: 12px 14px;
    }
    div.stButton > button {
        border-radius: 10px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.caption("GROUP LAMBDA  ·  CSE 3521")
st.title("Amazon food review sentiment")
st.write(
    "Enter a review to compare five trained models. Add a review summary if you "
    "have one, or try a sample to see how each model responds."
)

st.markdown("#### Try a sample review")
sample_columns = st.columns(4)
for column, sample in zip(sample_columns, SAMPLE_REVIEWS):
    with column:
        if st.button(
            sample["button"],
            key=f"sample_{sample['button']}",
            width="stretch",
        ):
            st.session_state["review_text"] = sample["text"]
            st.session_state["review_summary"] = sample["summary"]

with st.form("review_form"):
    review_text = st.text_area(
        "Review text",
        height=135,
        key="review_text",
        placeholder="What did the customer think about the product?",
    )
    review_summary = st.text_input(
        "Review summary (optional)",
        key="review_summary",
        placeholder="A short headline, if available",
    )
    analyze = st.form_submit_button("Analyze sentiment", type="primary")

predict_tab, comparison_tab, about_tab = st.tabs(
    ["Predictions", "Model comparison", "About"]
)

with predict_tab:
    if analyze:
        if not review_text.strip():
            st.warning("Please enter review text before analyzing.")
        else:
            with st.spinner("Loading models and analyzing the review..."):
                loaded_models = load_prediction_models()
                predictions = {}
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
                    predictions[model_key] = (
                        predicted_label,
                        positive_probability,
                    )

            positive_votes = sum(
                label == 1 for label, _ in predictions.values()
            )
            negative_votes = len(predictions) - positive_votes
            st.info(
                f"**Model agreement:** {positive_votes} of {len(predictions)} "
                f"predict positive; {negative_votes} predict negative."
            )

            for family in dict.fromkeys(model["family"] for model in MODELS):
                family_models = [
                    model for model in MODELS if model["family"] == family
                ]
                st.subheader(family)
                columns = st.columns(len(family_models))
                for column, config in zip(columns, family_models):
                    predicted_label, positive_probability = predictions[
                        config["key"]
                    ]
                    with column:
                        with st.container(border=True):
                            st.markdown(f"**{config['representation']}**")
                            st.metric(
                                "Positive probability",
                                f"{positive_probability:.1%}",
                            )
                            if predicted_label == 1:
                                st.success("Positive sentiment")
                                confidence = positive_probability
                            else:
                                st.error("Negative sentiment")
                                confidence = 1 - positive_probability
                            st.caption(f"Prediction confidence: {confidence:.1%}")
    else:
        st.info("Enter a review or choose a sample, then analyze its sentiment.")

with comparison_tab:
    st.subheader("Five saved models")
    st.dataframe(
        [
            {
                "Model family": model["family"],
                "Feature representation": model["representation"],
                "Summary repeats": model["summary_repeats"],
            }
            for model in MODELS
        ],
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "Every model uses its own saved vectorizer and training-time summary format."
    )

with about_tab:
    st.subheader("How it works")
    st.write(
        "Each saved model transforms the review with its paired vectorizer, then "
        "estimates positive or negative sentiment. When supplied, the review "
        "summary is repeated using the same format as that model's training data."
    )
    st.caption(
        "These are model predictions, not a substitute for reading customer feedback."
    )
