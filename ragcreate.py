import chromadb
from llama_index.core import Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

# print("success")
llm = None
Settings.llm = Ollama(model="llama3.2:3b", request_timeout=360.0)

chroma_client = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = chroma_client.create_collection("mgs636test")

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")
Settings.embed_model = embed_model

documents = SimpleDirectoryReader("./data/").load_data()

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, embed_model=embed_model)

query_engine = index.as_query_engine(llm=Settings.llm)
response = query_engine.query("What topics are covered in this class?")
print(response)
