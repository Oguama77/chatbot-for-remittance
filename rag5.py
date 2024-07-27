import os
import streamlit as st
import openai
from dotenv import load_dotenv
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

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=st.secrets.OPENAI_API_KEY)

# Load PDF and preprocess
FILE_PATHS = ["documents/doc1.pdf","documents/doc2.pdf","documents/doc3.pdf","documents/doc4.pdf"]
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
st.title("Hamoye Remittance Chatbot")

st.write("This app is designed to answer questions about remittance")

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

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []  # Store the chat history

# Display chat history
def display_chat_history():
    for i, (user_message, bot_response) in enumerate(st.session_state['chat_history']):
        with st.expander(f"Question {i+1}", expanded=True):
            st.markdown(
                f"""
                <div style="background-color:#f0f0f5;padding:10px;border-radius:5px;color:#000000;">
                    <strong>User:</strong> {user_message}
                </div>
                """,
                unsafe_allow_html=True
            )
            st.write(f"**Bot:** {bot_response}")

# Function to clear chat history
def clear_chat_history():
    st.session_state['chat_history'] = []

# Sidebar for chat history and clear button
with st.sidebar:
    st.header("Previous Questions")
    # Display previous questions only
    for i, (user_message, _) in enumerate(st.session_state["chat_history"]):
        st.write(f"**{i + 1}.** {user_message}")

    # Add a button to clear chat history
    if st.button("Clear Chat History"):
        clear_chat_history()
        st.write("Chat history cleared.")


# Chat interface
prompt = st.chat_input("Ask a question about remittance:")

display_chat_history()

st.write("---")  # Divider lin

if prompt:
    st.write(f"**User:** {prompt}")
    answer = answer_question(prompt, st.session_state['session_id'])
    st.write("**HRC:**", answer)
    st.session_state['chat_history'].append((prompt, answer))  # Save Q&A in chat history