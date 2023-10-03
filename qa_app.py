import os
import PyPDF2
import random
import itertools
import streamlit as st
import pymongo

from io import StringIO
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.retrievers import SVMRetriever
from langchain.chains import QAGenerationChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.base import CallbackManager
from langchain.embeddings import HuggingFaceEmbeddings
from streamlit_option_menu import option_menu
from pymongo import MongoClient
from gridfs import GridFS

st.set_page_config(page_title="Quran GPT", page_icon="📖")

cluster = MongoClient("mongodb+srv://ifhamansari:maa55208@cluster0.ghvyhyq.mongodb.net/myDatabase?retryWrites=true&w=majority")
db = cluster["test"]
collection = db["student"]

# Navigation menu

selected = option_menu(
    menu_title= None,
    options=["Quran GPT 3.5", "Quran GPT 4"],
    icons= ["hypnotize", "bar-chart-fill"],
    orientation="horizontal",
)

if selected == "Quran GPT 4":
    st.warning("This is close due to mantenance 🙂")

# Initialize a dictionary to store chat histories
if 'chat_histories' not in st.session_state:
    st.session_state.chat_histories = {}

# Check if the chat ID exists in the session state, and generate a new one if not
if 'chat_id' not in st.session_state:
    st.session_state.chat_id = str(random.randint(1000000, 9999999))

# Function to create a new chat instance
def create_new_chat():
    chat_id = str(random.randint(1000000, 9999999))
    return {
        'user_questions': [],
        'answers': [],
        'chat_id': chat_id,
    }


def view_file(file_id):
    fs = GridFS(db)
    file = fs.get(file_id)
    if file:
        st.write("File Name:", file.filename)

        # Display file content with a maximum height and scrollbar
        st.markdown(
            f'<div style="max-height: 90vh; overflow-y: auto; background-color: #f8f9fa; padding: 20px; margin-bottom: 12px; border-radius: 10px; color: #000;">{file.read().decode("utf-8")}</div>',
            unsafe_allow_html=True
        )

        st.session_state.file_being_viewed = True

# Function to display a chat in the sidebar
def display_chat(chat_id):
    chat_instance = st.session_state.chat_histories.get(chat_id)
    if chat_instance is not None:
        st.sidebar.subheader(f"Chat {chat_instance['chat_id']}")
        for i, (question, answer) in enumerate(zip(chat_instance['user_questions'], chat_instance['answers'])):
            st.sidebar.write(f"User : {question}")
            st.sidebar.write(f"Quran Gpt: {answer}")

        # if 'file_id' in chat_instance:
        #     if st.button("View File"):
        #         view_file(chat_instance['file_id'])
        #         if st.button("Close File"):
        #             st.text("File closed.")
        #     else:
        #         if st.button("Close File"):
        #             st.text("File closed.")

def store_file(file_data):
    fs = GridFS(db)
    file_id = fs.put(file_data['file_content'], filename=file_data['file_name'])
    return file_id

@st.cache_data
def load_docs(files):
    st.info("`Reading doc ...`")
    all_text = ""
    for file_path in files:
        file_extension = os.path.splitext(file_path.name)[1]
        if file_extension == ".pdf":
            pdf_reader = PyPDF2.PdfReader(file_path)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            all_text += text
        elif file_extension == ".txt":
            stringio = StringIO(file_path.getvalue().decode("utf-8"))
            text = stringio.read()
            all_text += text
        else:
            st.warning('Please provide txt or pdf.', icon="⚠️")
    return all_text

@st.cache_resource
def create_retriever(_embeddings, splits, retriever_type):
    if retriever_type == "SIMILARITY SEARCH":
        try:
            vectorstore = FAISS.from_texts(splits, _embeddings)
        except (IndexError, ValueError) as e:
            st.error(f"Error creating vectorstore: {e}")
            return
        retriever = vectorstore.as_retriever(k=5)
    elif retriever_type == "SUPPORT VECTOR MACHINES":
        retriever = SVMRetriever.from_texts(splits, _embeddings)

    return retriever

@st.cache_resource
def split_texts(text, chunk_size, overlap, split_method):
    # Split texts
    # IN: text, chunk size, overlap, split_method
    # OUT: list of str splits

    st.info("`Splitting doc ...`")

    split_method = "RecursiveTextSplitter"
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap)

    splits = text_splitter.split_text(text)
    if not splits:
        st.error("Failed to split document")
        st.stop()

    return splits

def main():
    foot = f"""
    <div style="
        position: fixed;
        bottom: 0;
        left: 30%;
        right: 0;
        width: 50%;
        padding: 0px 0px;
        text-align: center;
    ">
    </div>
    """
    st.markdown(foot, unsafe_allow_html=True)
    
    # Add custom CSS
    st.markdown(
        """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .css-card {
            border-radius: 0px;
            padding: 30px 10px 10px 10px;
            background-color: #f8f9fa;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 10px;
            font-family: "IBM Plex Sans", sans-serif;
        }
        
        .card-tag {
            border-radius: 0px;
            padding: 1px 5px 1px 5px;
            margin-bottom: 10px;
            position: absolute;
            left: 0px;
            top: 0px;
            font-size: 0.6rem;
            font-family: "IBM Plex Sans", sans-serif;
            color: white;
            background-color: green;
        }
        
        .css-zt5igj {left:0;}
        span.css-10trblm {margin-left:0;}
        div.css-1kyxreq {margin-top: -40px;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.image("img/quran-logo.png")
   
    st.write(
    f"""
    <div style="display: flex; align-items: center; margin-left: 0;">
    </div>
    """,
    unsafe_allow_html=True,
        )
    
    st.sidebar.title("Menu")
    
    embedding_option = st.sidebar.radio(
        "Choose Embeddings", ["OpenAI Embeddings"])
    retriever_type = st.sidebar.selectbox(
        "Choose Retriever", ["SIMILARITY SEARCH", "SUPPORT VECTOR MACHINES"])

    # Use RecursiveCharacterTextSplitter as the default and only text splitter
    splitter_type = "RecursiveCharacterTextSplitter"

    # Check if the "New Chat" button is clicked
    if st.sidebar.button("New Chat", key="new_chat_button"):  
        # Generate a unique chat ID
        chat_id = str(random.randint(1000000, 9999999))
        # Create a new chat instance and add it to the chat_histories dictionary
        st.session_state.chat_histories[chat_id] = create_new_chat()
        st.session_state.chat_histories[chat_id]['answers'] = []

    # Display existing chats in the sidebar
    chat_ids = list(st.session_state.chat_histories.keys())
    selected_chat_id = st.sidebar.selectbox("Select Chat", chat_ids)
    # Check if the selected chat ID is valid
    if selected_chat_id:
        if selected_chat_id in chat_ids:
            current_chat = st.session_state.chat_histories[selected_chat_id]
            # Display and interact with the chat
            display_chat(selected_chat_id)
        else:
            st.warning("Selected chat not found.")
       

    # Check if the API key is provided
    if 'openai_api_key' not in st.session_state:
        openai_api_key = st.text_input(
            'Please enter your OpenAI API key or [get one here](https://platform.openai.com/account/api-keys)', value="", placeholder="Enter the OpenAI API key which begins with sk-")
        if openai_api_key:
            st.session_state.openai_api_key = openai_api_key
            os.environ["OPENAI_API_KEY"] = openai_api_key
        else:
            return

    uploaded_files = st.file_uploader("Upload a PDF or TXT Document", type=[
                                      "pdf", "txt"], accept_multiple_files=True)

    if uploaded_files:
        # Check if last_uploaded_files is not in session_state or if uploaded_files are different from last_uploaded_files
        if 'last_uploaded_files' not in st.session_state or st.session_state.last_uploaded_files != uploaded_files:
            st.session_state.last_uploaded_files = uploaded_files
            if 'eval_set' in st.session_state:
                del st.session_state['eval_set']

        # Load and process the uploaded PDF or TXT files.
        loaded_text = load_docs(uploaded_files)
        st.write("Documents uploaded and processed.")

        # Store the file in the database
        file_data = {
            "file_name": uploaded_files[0].name,
            "file_content": uploaded_files[0].read()  # Store the file content
        }
        file_id = store_file(file_data)

        # Split the document into chunks
        splits = split_texts(loaded_text, chunk_size=1000,
                             overlap=0, split_method=splitter_type)

        # Display the number of text chunks
        num_chunks = len(splits)
        st.write(f"Number of text chunks: {num_chunks}")

        st.write("File uploaded. Click below to view the file:")
        if st.button("View File",  key="view_file_button"):
            view_file(file_id)
            st.button("Close File")

        # Embed using OpenAI embeddings
        # Embed using OpenAI embeddings or HuggingFace embeddings
        if embedding_option == "OpenAI Embeddings":
            embeddings = OpenAIEmbeddings()
        elif embedding_option == "HuggingFace Embeddings(slower)":
            # Replace "bert-base-uncased" with the desired HuggingFace model
            embeddings = HuggingFaceEmbeddings()

        retriever = create_retriever(embeddings, splits, retriever_type)

        # Initialize the RetrievalQA chain with streaming output
        callback_handler = StreamingStdOutCallbackHandler()
        callback_manager = CallbackManager([callback_handler])

        chat_openai = ChatOpenAI(
            streaming=True, callback_manager=callback_manager, verbose=True, temperature=0)
        qa = RetrievalQA.from_chain_type(
            llm=chat_openai, retriever=retriever, chain_type="stuff", verbose=True)

        # Question and answering
        user_question = st.text_input("Enter your question:")
        if user_question:
            # Find the latest chat instance
            current_chat = st.session_state.chat_histories[selected_chat_id]

            # Add the user question to the current chat instance
            if user_question not in current_chat['user_questions']:
                current_chat['user_questions'].append(user_question)
                answer = qa.run(user_question)
                current_chat['answers'].append(answer)

                # Link the file to the chat by associating its ID with the chat
                current_chat['file_id'] = file_id

                # Database
                # Create the post object with the incremented ID
                post = {
                    'user-question': user_question,
                    'Answer': answer,
                    'file_id': file_id
                }
                collection.insert_one(post)

                st.session_state.chat_histories[selected_chat_id] = current_chat

selected_chat_id = None  # Define selected_chat_id at the beginning

if __name__ == "__main__":
    main()

# Move the display_chat call outside the main function
if selected_chat_id is not None:
    display_chat(selected_chat_id)