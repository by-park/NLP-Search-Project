# Python Packages
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceBgeEmbeddings

# Load vectorstore
# refer to 'save_vectorstore.py'
model_name = "BAAI/bge-large-en-v1.5"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {"normalize_embeddings": True}

embedding_function = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

db = FAISS.load_local('faiss_store', embedding_function, allow_dangerous_deserialization=True)

# Query the vector database
query="Show me the transformer class configuration"
matched_docs = db.similarity_search(query=query, k=5)
print(matched_docs)