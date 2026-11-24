import streamlit as st

# Set page title and layout
st.set_page_config(page_title="Amazon Review Sentiment", layout="wide")

st.title("Amazon Food Review Sentiment")
st.caption("small grey helper text")
st.write("plain text or markdown")

# Inputs. The returned value is whatever the user currently typed.
text = st.text_area("Review text", height=150, key="review_text")
summary = st.text_input("Review summary (optional)", key="summary")

# A button returns True only on the run right after it is clicked.
clicked = st.button("Analyze sentiment", type="primary")

# Layout
tab1, tab2, tab3 = st.tabs(["Predict", "Model comparison", "About"])
with tab1:
    st.write("content of the first tab")

col_a, col_b, col_c = st.columns(3)
with col_a:
    st.subheader("Naive Bayes")

# Showing results
st.metric("P(positive)", "97.5%")
st.success("Positive")      # green box
st.error("Negative")        # red box
st.warning("Please enter a review")
st.info("note")
st.dataframe(some_pandas_dataframe)
st.bar_chart(some_pandas_dataframe)
st.image("report/screenshots/example.png")

with st.expander("How it works"):
    st.write("hidden until clicked")

with st.spinner("Loading models..."):
    pass