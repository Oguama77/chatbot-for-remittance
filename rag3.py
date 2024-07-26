import os
import streamlit as st
import openai
#from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Setting the OpenAI API key 

#load_dotenv()
#OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=st.secrets.OPENAI_API_KEY)

# Load PDF and preprocess
FILE_PATHS = ["documents/migration_development_brief_38_june_2023_0.pdf"]
all_documents = []

for file_path in FILE_PATHS:
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    all_documents.extend(pages)

# Initialize text splitter and split documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(all_documents)

# Initialize embeddings and vector store
vectorstore = FAISS.from_documents(documents=splits, embedding=OpenAIEmbeddings(openai_api_key=st.secrets.OPENAI_API_KEY))
retriever = vectorstore.as_retriever()

# Contextualize question prompt
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# Question-answering system prompt
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# Create retrieval-augmented generation (RAG) chain
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Statefully manage chat history
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
) 

# Streamlit app setup
st.title("Remittance Chatbot")

st.write("This app allows you to ask questions about remittance using RAG")

def answer_question(prompt, session_id):
    # Update: Pass the session_id in config
    result = conversational_rag_chain.invoke(
        {"input": prompt},
        config={"configurable": {"session_id": session_id}}
    )
    return result['answer']  # Extract the answer from the result

# Streamlit app logic
if 'session_id' not in st.session_state:
    st.session_state['session_id'] = str(os.urandom(16))  # Unique session ID for chat history

prompt = st.chat_input("Ask a question about remittance:")

if prompt:
    st.write(f"**Message by user:** {prompt}")
    answer = answer_question(prompt, st.session_state['session_id'])
    st.write("**Answer:**", answer)
else:
    st.write("Please enter a question.")
