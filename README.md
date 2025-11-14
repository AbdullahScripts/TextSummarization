## Text Summarization using Transformer Models

This project implements an **Abstractive Text Summarization System** using state-of-the-art **Transformer models** (BART, T5, BERT2BERT).  
It generates **concise, human-like summaries** from long articles, blogs, and news reports.  
The model is fine-tuned on the **CNN/Daily Mail dataset** for high-quality performance.



## 🚀 Features

- Abstractive summarization with HuggingFace Transformers  
- Fine-tuned on CNN/Daily Mail  
- Clean preprocessing & tokenization pipeline  
- Multiple supported models (BART, T5, PEGASUS, BERT2BERT)  
- Easy inference for custom text summaries  
- Jupyter Notebook implementation  

---

## 📂 Dataset

**CNN/Daily Mail Summarization Dataset**  
🔗 https://huggingface.co/datasets/cnn_dailymail

```python
from datasets import load_dataset
dataset = load_dataset("cnn_dailymail", "3.0.0")
````

---

## 🧠 Model Overview

### ✔ BART (facebook/bart-large-cnn) – Best for summarization

### ✔ T5 – Text-to-text transformer

### ✔ BERT2BERT – Encoder-decoder model

Fine-tuning includes:

* Cross-entropy loss
* Teacher forcing
* Attention masking

---

## 📘 How It Works

1. Load & preprocess dataset
2. Tokenize input/output sequences
3. Initialize transformer model
4. Fine-tune on article–summary pairs
5. Generate summaries for custom text

Example:

```python
inputs = tokenizer([text], max_length=1024, truncation=True, return_tensors="pt")
summary_ids = model.generate(inputs["input_ids"], max_length=150, min_length=40)
print(tokenizer.decode(summary_ids[0], skip_special_tokens=True))
```

---

## ▶ Run the Project

```bash
git clone https://github.com/AbdullahScripts/TextSummarization.git
cd TextSummarization
pip install -r requirements.txt
jupyter notebook
```

Open **Text_summarization.ipynb** and run the cells.

---

## 📊 Output Example

**Input:** 300–500 word article
**Output:** 2–4 sentence concise summary with high coherence.

---

## 🔮 Future Enhancements

* Add ROUGE/BLEU evaluation
* Deploy with Gradio / Streamlit
* Add REST API endpoint

---

## 📜 License

MIT License
