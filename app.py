import streamlit as st
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Set page config
st.set_page_config(page_title="Parliament Question Subject Classifier", layout="centered")
st.title("🗳️ Parliament Question ➜ Subject Classifier")

st.markdown("""
Enter a **parliamentary question** below and the app will classify it into a **subject category**  
based on your RNN model trained on Parliament Debates 🇮🇳.
""")

# Load tokenizer
with open("model/subject_tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load label encoder
with open("model/subject_label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Load trained model
model = load_model("model/subject_rnn_model.h5")

# Input
question = st.text_area("📝 Enter Parliament Question", placeholder="e.g. What measures has the government taken to address rising unemployment?")

if st.button("🔍 Predict Subject"):
    if not question.strip():
        st.warning("Please enter a valid question.")
    else:
        # Tokenize and pad
        seq = tokenizer.texts_to_sequences([question])
        padded = pad_sequences(seq, maxlen=50, padding='post')

        # Predict
        pred = model.predict(padded)[0].argmax()
        subject = label_encoder.inverse_transform([pred])[0]

        st.success(f"✅ **Predicted Subject:** `{subject}`")
