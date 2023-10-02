import pandas as pd
import os
import sys
import openai
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, GPT4AllEmbeddings
from langchain.vectorstores import Chroma, FAISS
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader, SitemapLoader
from tqdm import tqdm
from constant import *

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 10
MODEL = "gpt-3.5-turbo"
TEMPERATURE = 0.9
HEADER ="You are a helpful assistant.given answer only based on given documents. Consider full data, do not work with sample data and dont costomize answer if same question is there. Do not answer any question not about the info. Never break character."

class Test():

    def __init__(self) -> None:
        # self.read_files()
        os.environ["OPENAI_API_KEY"] = API_KEY
        self.qa_chain = self.chains()
        

    def embed(self):
        
        loader = SitemapLoader(SITEMAP)
        excel_data = loader.load()
        excel_document = excel_data
        # print(excel_document)
        print("loader done 1..........")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        print("CharacterTextSplitter done 1..........")
        excel_doc = text_splitter.split_documents(excel_document)
        print(excel_doc,"_>>>>>>>>>>>")
        vectorestores = FAISS.from_documents(excel_doc, embedding=OpenAIEmbeddings())
        print("vectorestores done..........")
            # vectorestores.persist()
        return vectorestores
    
    def chains(self):
        vectorestore = self.embed()
        # memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        qa_chain = ConversationalRetrievalChain.from_llm(llm=ChatOpenAI(model=MODEL, temperature=TEMPERATURE), retriever=vectorestore.as_retriever(), verbose=False)
        return qa_chain
    
    def run(self,query):
        question = f"{HEADER}" + f"{query}" 
       
        answer = (self.qa_chain({"question": question, "chat_history":''}))
        return answer["answer"]   

if __name__=="__main__":
    v= Test()
    while True:
            print("\nWelcome to descriptive Recommendation System")
            question = input("Type discription here : -")
            if question in ["quit", "q"]:
                    sys.exit()
            print("Processing...")
            answer = v.run(question)
            print("\nAnswer: ", answer)