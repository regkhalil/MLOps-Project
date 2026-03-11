"""Streamlit UI for 20 Newsgroups text classification."""

import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

CATEGORIES = {
    "alt.atheism": "Atheism & Secularism",
    "comp.graphics": "Computer Graphics",
    "comp.os.ms-windows.misc": "Windows OS",
    "comp.sys.ibm.pc.hardware": "PC Hardware",
    "comp.sys.mac.hardware": "Mac Hardware",
    "comp.windows.x": "X Window System",
    "misc.forsale": "For Sale",
    "rec.autos": "Automobiles",
    "rec.motorcycles": "Motorcycles",
    "rec.sport.baseball": "Baseball",
    "rec.sport.hockey": "Hockey",
    "sci.crypt": "Cryptography",
    "sci.electronics": "Electronics",
    "sci.med": "Medicine & Health",
    "sci.space": "Space & Astronomy",
    "soc.religion.christian": "Christianity",
    "talk.politics.guns": "Gun Politics",
    "talk.politics.mideast": "Middle East Politics",
    "talk.politics.misc": "General Politics",
    "talk.religion.misc": "Religion & Beliefs",
}

st.set_page_config(page_title="20 Newsgroups Classifier", layout="wide")
st.title("20 Newsgroups Text Classifier")

left, right = st.columns([1, 2])

# Process the right column first so session_state is updated before rendering categories
with right:
    st.subheader("Classify Text")
    text = st.text_area(
        "Enter text to classify (max 1000 characters):",
        height=200,
        max_chars=1000,
    )

    if st.button("Classify", type="primary"):
        if not text.strip():
            st.warning("Please enter some text.")
        else:
            try:
                resp = requests.post(
                    f"{API_URL}/predict",
                    json={"text": text},
                    timeout=30,
                )
                resp.raise_for_status()
                result = resp.json()
                st.session_state["predicted_label"] = result["label"]
                st.success(f"**Predicted category:** {result['display_name']}")
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the classification API.")
            except requests.exceptions.HTTPError as e:
                st.error(f"API error: {e}")

with left:
    st.subheader("Categories")
    predicted_label = st.session_state.get("predicted_label")
    for key, display in CATEGORIES.items():
        if key == predicted_label:
            st.markdown(f"**:green[► {display}]**")
        else:
            st.markdown(f"- {display}")
