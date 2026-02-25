import os
import streamlit as st
import time

from dotenv import load_dotenv
load_dotenv()

st.write("API Key Loaded:", os.getenv("OPENAI_API_KEY") is not None)

# LLM
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Vector Store
from langchain_community.vectorstores import FAISS

# Document Loader
from langchain_community.document_loaders import UnstructuredURLLoader

# Text Splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter


from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

llm = ChatOpenAI(model="gpt-4o-mini")

st.title("Equity Research Analyst: News Research Tool")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    if url:
        urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")

#for UI
main_placeholder = st.empty()

if process_url_clicked and urls:
    #load data
    loaders = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading......✅✅✅")
    data = loaders.load()

    #split data
    # Splitting data into chunks

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = text_splitter.split_documents(data)

    # Creating vector embeddings for these chunks and save them to FAISS index

    # Creating embeddings of the chunks using openAIEmbeddings
    embeddings = OpenAIEmbeddings()

    # Directory to store FAISS index
    index_path = "vector_index"

    # Passing the documents and embeddings inorder to create FAISS vector index and save, if index does NOT exist → create and save
    if not os.path.exists(index_path):
        vectorindex_openai = FAISS.from_documents(docs, embeddings)
        vectorindex_openai.save_local(index_path)
    else:
        # Load existing index
        vectorindex_openai = FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True
        )

    main_placeholder.text("Embedding Vectors......✅✅✅")


#Query

query = st.text_input("Question: ")

if query and os.path.exists("vector_index"):

    embeddings = OpenAIEmbeddings()

    vectorIndex = FAISS.load_local("vector_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vectorIndex.as_retriever()

    prompt = ChatPromptTemplate.from_template(
        """
        Answer the question using only the provided context.
        If the answer is not found, say you don't know.

        Context:
        {context}

        Question:
        {question}
        """
    )

    rag_chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
    )

    with st.spinner("Generating answer..."):
        response = rag_chain.invoke(query)

    st.header("Answer")
    st.write(response.content)


    docs = retriever.invoke(query)

    if docs:
        st.subheader("Sources:")
        for doc in docs:
            source = doc.metadata.get("source", "Unknown source")
            st.write(source)