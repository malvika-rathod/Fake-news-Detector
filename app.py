# === MUST BE FIRST STREAMLIT COMMAND ===
import streamlit as st
st.set_page_config(page_title="Fake News Detector", layout="centered")

# === Imports ===
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
import torch
import torch.nn.functional as F
import pandas as pd
from streamlit_lottie import st_lottie
import json
import os
from datasets import Dataset
import tempfile
import shutil
import requests
import matplotlib.pyplot as plt

# === Load Animation ===
def load_lottiefile(path: str):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None

header_anim = load_lottiefile("animations/header.json")

# === Load Model & Tokenizer ===
def load_model():
    model_path = "bert_model"
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

model, tokenizer = load_model()

# === Session State ===
if 'pred_history' not in st.session_state:
    st.session_state.pred_history = []
if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = {}

# === Dark Mode ===
mode = st.toggle("🌗 Dark Mode", value=False)
if mode:
    st.markdown("""
        <style>
        body { background-color: #121212; color: #f0f0f0; }
        .stTextArea textarea { background-color: #2e2e2e; color: white; }
        </style>
    """, unsafe_allow_html=True)

# === Header ===
if header_anim:
    st_lottie(header_anim, height=130, key="header")
st.title("📰 Fake News Detector (BERT-Based)")
st.caption("Classify news articles as REAL or FAKE using a fine-tuned BERT model.")

# === Sample Dropdown ===
sample_news = {
    "Select an example...": "",
    "REAL: Chandrayaan-3 Moon Landing": "India's Chandrayaan-3 successfully landed on the Moon’s south pole, becoming the first country to do so.",
    "FAKE: NASA confirms 6 days of darkness": "NASA confirmed Earth will experience 6 days of darkness due to planetary alignment.",
}
choice = st.selectbox("📚 Load a sample article:", list(sample_news.keys()))
def_input = sample_news[choice]

# === User Input ===
user_input = st.text_area("✍️ Paste or edit your news article:", value=def_input, height=200)

# === Reference Mapping ===
reference_links = {
    "India's Chandrayaan-3 successfully landed on the Moon’s south pole, becoming the first country to do so.": "https://www.isro.gov.in/Chandrayaan3.html",
    "The World Health Organization declared the end of the global COVID-19 emergency after more than three years. The organization emphasized the importance of ongoing surveillance and vaccinations to prevent future outbreaks.": "https://www.who.int/news/item/05-05-2023-statement-on-the-15th-meeting-of-the-ihr-emergency-committee-regarding-the-covid-19-pandemic"
}

# === Reference Suggestion (Bing News Search API or placeholder) ===
def suggest_reference(query):
    try:
        search_url = f"https://www.bing.com/news/search?q={query}&format=json"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(search_url, headers=headers)
        if resp.ok:
            return f"https://www.bing.com/news/search?q={query.replace(' ', '+')}"
    except:
        pass
    return None

# === Check for prior correction ===
def load_feedback_dict():
    if os.path.exists("feedback.csv"):
        df = pd.read_csv("feedback.csv", names=["text", "label"])
        return {row["text"].strip(): int(row["label"]) for _, row in df.iterrows()}
    return {}

feedback_dict = load_feedback_dict()

# === Predict ===
if st.button("🔍 Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text first.")
    else:
        clean_input = user_input.strip()

        is_corrected = False
        if clean_input in feedback_dict:
            corrected_label = feedback_dict[clean_input]
            label = "REAL" if corrected_label == 1 else "FAKE"
            confidence_pct = 100.0
            st.warning("Corrected from user feedback")
            is_corrected = True
        else:
            inputs = tokenizer(clean_input, return_tensors="pt", truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)
            confidence, prediction = torch.max(probs, dim=1)
            corrected_label = prediction.item()
            label = "REAL" if corrected_label == 1 else "FAKE"
            confidence_pct = round(confidence.item() * 100, 2)

        st.subheader("🔎 Prediction")
        if label == "FAKE":
            st.error(f"**{label}** (Confidence: **{confidence_pct}%**)")
        else:
            st.success(f"**{label}** (Confidence: **{confidence_pct}%**)")
            if clean_input in reference_links:
                st.markdown(f"🔗 [Read source]({reference_links[clean_input]})")
            else:
                suggested_link = suggest_reference(clean_input)
                if suggested_link:
                    st.markdown(f"🔍 Suggested link: [Search on Bing]({suggested_link})")

        st.session_state.last_prediction = {
            "Text": clean_input,
            "Label": label,
            "Predicted": corrected_label,
            "Confidence": confidence_pct
        }
        st.session_state.pred_history.append({
            "Text": clean_input[:40] + "...",
            "Label": label,
            "Confidence": confidence_pct
        })

# === Confidence History Chart ===
if st.session_state.pred_history:
    st.subheader("📈 Confidence Over Time")
    df = pd.DataFrame(st.session_state.pred_history)
    fig, ax = plt.subplots()
    colors = df["Label"].map({"REAL": "green", "FAKE": "red"})
    ax.bar(df["Text"], df["Confidence"], color=colors)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Article Snippet")
    plt.ylabel("Confidence (%)")
    plt.tight_layout()
    st.pyplot(fig)
