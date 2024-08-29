from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_elasticsearch import ElasticsearchStore
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

EMBEDDING_MODEL_NAME = "thenlper/gte-small"

template="""You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use five sentences minimum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""

if __name__ == '__main__':
    loader = TextLoader(r"C:\Users\Suraj\Desktop\Python\elastic-search-rag\test.txt")

    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    docs = text_splitter.split_documents(documents=documents)
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        multi_process=True,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},  # Set `True` for cosine similarity
    )

    # vector_db = ElasticsearchStore.from_documents(
    #     docs,
    #     embedding=embeddings,
    #     index_name="test",
    #     es_cloud_id=os.getenv("ES_CLOUD_ID"),
    #     es_api_key=os.getenv("ES_API_KEY")
    # )

    vector_db = ElasticsearchStore(
        embedding=embeddings,
        index_name="test",
        es_cloud_id=os.getenv("ES_CLOUD_ID"),
        es_api_key=os.getenv("ES_API_KEY")
    )

    retriever = vector_db.as_retriever()

    prompt = ChatPromptTemplate.from_template(template)

    llm = model = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="mixtral-8x7b-32768",
    )
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    query = "Who is the king john?"
    result = rag_chain.invoke(query)
    print(result)


