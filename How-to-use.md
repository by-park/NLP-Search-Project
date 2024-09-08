# How to use
1. Intall python packages
```bash
pip install -r requirements.txt
```
> ※ Intalled package list: datasets langchain langchain_community sentence_transformers faiss-cpu beautifulsoup4

2. Load and Ask query to saved vectorstore
```bash
python ./load_vectorstore.py
```
> ※ Huggingface Documents are stored in vectorstore by `save_vectorstore.py`