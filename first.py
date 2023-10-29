import streamlit as st
from langchain.llms import HuggingFaceHub
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import faiss
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import openai
from langchain.callbacks import get_openai_callback
import os
import PyPDF2
import random
import itertools
import streamlit as st
from io import StringIO
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.retrievers import SVMRetriever
from langchain.chains import QAGenerationChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.embeddings import HuggingFaceEmbeddings


def Open():
    embeddings = OpenAIEmbeddings()
    document = faiss.FAISS.from_texts(chunks, embeddings)

    user_question = st.text_input("Ask a question about your PDFs:")
    if user_question:
        docs = document.similarity_search(user_question)

        llm = openai.OpenAI()
        chain = load_qa_chain(llm, chain_type="stuff")
            # with get_openai_callback() as cb:
        response = chain.run(input_documents=docs, question=user_question)
                # print(cb)
        st.write(response)

        embeddings = HuggingFaceEmbeddings()
        document = faiss.FAISS.from_texts(chunks, embeddings)


def hug():
     embeddings = OpenAIEmbeddings()
     document = faiss.FAISS.from_texts(chunks, embeddings)

    user_question = st.text_input("Ask a question about your PDFs:")
    if user_question:
        docs = document.similarity_search(user_question)
            
        llm = openai.OpenAI()
        chain = load_qa_chain(llm, chain_type="stuff")
            #with get_openai_callback() as cb:
        response = chain.run(input_documents=docs, question = user_question)
                #print(cb)
        st.write(response) 















def main():
    load_dotenv()
    st.set_page_config(page_title="Eva", page_icon="👾", layout="wide")

    st.header("Chat 💬 With Your Pdf With EVA👾")
    with st.sidebar:
        st.markdown("<h1 style='text-align: center'>👾EVA</h1>",
                    unsafe_allow_html=True)
        #uploading file 
        pdf = st.file_uploader("Upload Your Document Here", type="pdf")
    
    embedding_option = st.sidebar.radio(
        "Choose Embeddings", ["OpenAI Embeddings", "HuggingFace Embeddings(slower)"])


    #extracting the file 
    if pdf is not None:
        pdf_reader =PdfReader(pdf)
        text =""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # spliting text into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size =1000,
            chunk_overlap =200,
            length_function =len
        )
        chunks =text_splitter.split_text(text)
        



    
        
        if embedding_option == "OpenAI Embeddings":
            embeddings = Open()
        elif embedding_option =="HuggingFace Embeddings(slower)":
            embeddings = hug()


     
   




if __name__ == '__main__':
    main()