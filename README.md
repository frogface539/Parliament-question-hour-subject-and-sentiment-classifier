# 🇮🇳 Parliament Subject Classifier 🧠📜

A deep learning project using **RNN (SimpleRNN)** to classify Indian Parliament questions into their respective **subjects**. Built using TensorFlow/Keras and trained on a custom preprocessed dataset from Indian parliamentary proceedings.

---

## 🚀 Features

- 🔤 Classifies text-based parliamentary questions into subjects (e.g., Agriculture, Labour, Education, etc.)
- 🔡 Uses RNN architecture for sequence modeling
- 🧼 Includes preprocessing, cleaning, and label encoding
- 📈 Graphs show training vs validation performance
- 📂 Organized with proper `data/`, `model/`, and `notebooks/` folders

---

## 🧠 Model Details

- Architecture: `Embedding ➝ SimpleRNN ➝ Dropout ➝ Dense`
- Optimizer: Adam
- Loss: Sparse Categorical Crossentropy
- Epochs: 10–30 (experimented with)
- Accuracy: ~78% training accuracy, validation accuracy fluctuates (due to class imbalance)

---

## 📁 Folder Structure

```
Parliament Subject Classifier/
├── data/
│   ├── clean\_subject\_data.csv
│   └── TCPD\_QH.tsv
├── model/
│   ├── subject\_rnn\_model.h5
│   ├── subject\_tokenizer.pkl
│   └── subject\_label\_encoder.pkl
├── Notebooks/
│   ├── subject\_preparation.ipynb
│   ├── rnn\_subject\_classifier.ipynb
│   └── subject\_classification.ipynb
├── requirements.txt
└── README.md
````

---

## ⚠️ Disclaimer

This is an **academic project** meant for **exploration and learning**.  
Due to the **limited dataset size (~1800 samples)** and **class imbalance**, the predictions may be inconsistent or biased toward dominant subjects.

---

## 🔮 Future Improvements

- ✅ Use pre-trained embeddings (e.g., GloVe, Word2Vec)
- ✅ Augment or collect more parliament question data
- ✅ Merge similar subjects into broader categories (e.g., "Education", "Economy", "Health")
- ✅ Use GRU / LSTM for improved sequential learning
- ✅ Add confidence scores / top-2 subject predictions
- ✅ Expand model to handle **multi-label** tagging

---

## 💻 How to Run

1. Clone the repo:
```bash
git clone https://github.com/your-username/parliament-subject-classifier.git
cd parliament-subject-classifier
````

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run training from Jupyter Notebooks or adapt to a Streamlit frontend.

---

## 📚 Dataset Source

* [Parliament Debates - Kaggle](https://www.kaggle.com/datasets/saurabhshahane/parliament-debates)
* CSV/TSV cleaned and tokenized in the notebook

---
## 📜 License

MIT © 2025 Lakshay Jain