
# Fake new detector

A brief description of what this project does and who it's for

This project can be used to detect whether a piece of information on the net is real or fake.

# 📰 Fake News Detector Using BERT & Streamlit

This project is a **Fake News Detection Web Application** that uses a fine-tuned BERT model to classify news articles as **real** or **fake**. The application is built with **Streamlit** for the frontend and includes real-time prediction, user feedback handling, and automatic model retraining for improved accuracy over time.

---

## 🚀 Features

- ✅ Classifies news articles as **Real** or **Fake**
- 🤖 Built using a **BERT-based NLP model** fine-tuned on a labeled news dataset
- 📈 Displays **prediction confidence**, **confusion matrix**, and **accuracy**
- 🧠 Accepts user feedback to **retrain the model automatically**
- 🔄 Self-improving model based on user corrections
- 🌐 Clean and interactive **Streamlit web interface**
- Provides reference of the news article from search engines or wesbites

---

## 🧠 Model Overview

The model is a fine-tuned version of **BERT (Bidirectional Encoder Representations from Transformers)**. It has been trained on a balanced dataset of real and fake news articles from Kaggle and achieves over **90% accuracy** on the test set.

---

# 📰 Fake News Detector Using BERT & Streamlit

This project is a **Fake News Detection Web Application** that uses a fine-tuned BERT model to classify news articles as **real** or **fake**. It features real-time classification, confidence scores, feedback-based retraining, and an intuitive Streamlit UI.

---

## 🚀 Features

- ✅ Predicts whether a news article is **Real** or **Fake**
- 🤖 Built using a **BERT-based model** fine-tuned on a labeled news dataset
- 📊 Displays **prediction confidence**, **confusion matrix**, and accuracy
- 🔁 Accepts **user feedback** and supports **automatic retraining**
- 🧠 Learns and improves from incorrect predictions
- 🎯 Clean and responsive **Streamlit interface**

---

## 🧠 Model Overview

We use a **pre-trained BERT model**, fine-tuned on a balanced dataset of real and fake news. With proper preprocessing and training on Colab, the model achieves **90%+ accuracy**.

---

## 🖥️ Tech Stack

- **Frontend:** Streamlit
- **Backend & ML:** Python, BERT (via HuggingFace Transformers)
- **Training Platform:** Google Colab
- **Deployment:** VS Code (local) or Streamlit Cloud
- **Logging:** TensorBoard

---

## 📁 Project Structure

```
FAKE-NEWS-DETECTOR/
│
├── animations/
│   └── header.json                  # Lottie animation for UI
│
├── bert_model/                      # Pretrained BERT model files
│   ├── config.json
│   ├── model.safetensors
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   ├── tokenizer.json
│   └── vocab.txt
│
├── logs/                            # Training logs for TensorBoard
│   └── events.out.tfevents...
│
├── app.py                           # Main Streamlit application
├── bert_model.zip                   # Zipped model for portability
├── feedback.csv                     # Stores user feedback for retraining
├── news_clean.csv                   # Cleaned dataset used for training
├── news.csv                         # Original raw dataset
├── news_clean.zip                   # Zipped cleaned data
├── news.zip                         # Zipped raw data
├── requirements.txt                 # Python dependencies
└── README.md                        # Project documentation
```

---

## 📊 Evaluation Metrics

- **Accuracy:** ~90%
- **Confusion Matrix:** Visualizes correct vs incorrect predictions
- **Precision / Recall / F1-Score:** Deeper insight into model performance

---

## 🧪 How to Run the App Locally

1. **Clone the repository:**

```bash
git clone https://github.com/malvika-rathod/Fake-news-Detector.git
cd fake-news-detector
```

2. **Install the dependencies:**

```bash
pip install -r requirements.txt
```

3. **Run the app:**

```bash
streamlit run app.py
```

---

## 🔁 Model Retraining Logic

Whenever a user flags a wrong prediction, the text and corrected label are saved to `feedback.csv`. During scheduled retraining, this data is included to improve the model’s future accuracy.

---

## 📸 Screenshots

### 🏠 Home Page
![image](https://github.com/user-attachments/assets/47a086bf-e833-4b80-8524-485323434776)


### 🧠 Prediction Output
Real
![image](https://github.com/user-attachments/assets/66c2aab4-7ca4-4b3e-b4db-16e0f4f59e35)
Fake
![image](https://github.com/user-attachments/assets/b9b4942e-15eb-4cc2-ae58-d08f46c32422)


### 📈 Confidence over time
![image](https://github.com/user-attachments/assets/18c88b77-4a0f-48e5-afcf-a8e2c81009a1)


---

## ✅ Sample Input & Output

**Input:**
```
"WHO declares end to global COVID-19 emergency as cases steadily decline worldwide."(Real)
```
```
"Scientists find a way to live without sleep using a secret Himalayan herb."(FAke)
```

**Output:**
```
Prediction: Real  
Confidence: 95.64%
```
```
Prediction: Fake  
Confidence: 99.59%
```

---

## 📄 License

This project is licensed under the **MIT License**.

---

## 🙌 Acknowledgements

- HuggingFace 🤗 Transformers
- Streamlit
- Kaggle Fake News Dataset

---

## 👩‍💻 Developed By

**Malvika Rathod**  
📧 malvikarathod112002@gmail.com  
🔗 [GitHub](https://github.com/malvika-rathod)
