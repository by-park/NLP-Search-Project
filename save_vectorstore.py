# Python Packages
from datasets import load_dataset
from langchain.document_loaders.recursive_url_loader import RecursiveUrlLoader
from bs4 import BeautifulSoup as Soup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import FAISS

# Load document
url = "https://huggingface.co/docs"

loader = RecursiveUrlLoader(
    url=url,
    extractor=lambda x: Soup(x, "html.parser").text
)

docs = loader.load()

# Create a vector database
# FAISS: in memory database
text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
docs = text_splitter.split_documents(documents=docs)

model_name = "BAAI/bge-large-en-v1.5"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {"normalize_embeddings": True}

embedding_function = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

db = FAISS.from_documents(
    docs,
    embedding_function
)

db.save_local("faiss_store")