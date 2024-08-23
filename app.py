from dotenv import load_dotenv
import os
import google.generativeai as genai
from youtube_transcript_api import YouTubeTranscriptApi
import streamlit as st
import streamlit.components.v1 as stc
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import pymongo
from PIL import Image
from PyPDF2 import PdfReader
from werkzeug.security import generate_password_hash, check_password_hash
from PIL import Image
import asyncio
import pandas as pd
from pandasai import Agent
# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Connect to MongoDB
client = pymongo.MongoClient(os.getenv("MONGODB_URI"))
db = client["chatbot_db"]
users_collection = db["chatbot_collection"]
queries_collection = db["queries_collection"]

# Function to get user ID from session
def get_user_id_from_session():
    return st.session_state.get("user_id", None)

# Function to save query and user ID to MongoDB
def save_query(user_id, query):
    queries_collection.insert_one({"user_id": user_id, "query": query})

# Function to retrieve all queries for a specific user from MongoDB
def get_user_queries(user_id):
    return [query["query"] for query in queries_collection.find({"user_id": user_id})]

# Function to delete a specific query for a user from MongoDB
def delete_user_query(user_id, query):
    queries_collection.delete_one({"user_id": user_id, "query": query})

# Function to display recent searches for a specific user in the sidebar along with delete buttons
def display_user_recent_searches(user_id):
    st.sidebar.subheader("Your Recent Searches")
    queries = get_user_queries(user_id)
    queries.reverse()
    if not queries:
        st.sidebar.write("No recent searches.")
    else:
        for index, query in enumerate(queries):
            button_key = f"button_{index}_{query}"
            delete_button_key = f"delete_button_{index}_{query}"

            query_col, delete_col = st.sidebar.columns([5, 1])
            with query_col:
                query_with_delete_button = st.empty()
                query_with_delete_button.write(query)
            with delete_col:
                delete_button_clicked = st.empty().button("❌", key=delete_button_key)

            if delete_button_clicked:
                delete_user_query(user_id, query)
                st.rerun()

            if query_with_delete_button.button(query, key=button_key):
                generate_answer(user_id, query)

# Function to generate an answer for the selected query
def generate_answer(user_id, query):
    answer = LLM_Response(user_id, query)
    st.subheader("Response:")
    response_text = "\n".join([word.text for word in answer])
    st.write(response_text)
    if query not in get_user_queries(user_id):
        save_query(user_id, query)

# Function to handle user signup
def signup():
    st.title("Sign Up")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")

    if st.button("Sign Up"):
        if password == confirm_password:
            # Corrected the hashing method
            hashed_password = generate_password_hash(password, method="pbkdf2:sha256")
            users_collection.insert_one({"username": username, "password": hashed_password})
            st.success("Sign up successful! Please log in.")
        else:
            st.error("Passwords do not match.")

# Function to handle user login
def login():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        user = users_collection.find_one({"username": username})
        if user and check_password_hash(user["password"], password):
            st.session_state.user_id = username
            st.success("Login successful!")
            # A workaround to "rerun" by setting query parameters
            st.query_params = {} 
            st.rerun() 
        else:
            st.error("Invalid username or password")


# Function to handle logout
def logout():
    if st.sidebar.button("Logout"):
        st.session_state.pop("user_id", None)
        st.rerun()

# Function to manage user sessions and authentication
def manage_user_session():
    user_id = get_user_id_from_session()
    if user_id is None:
        auth_choice = st.sidebar.selectbox("Choose an option", ["Login", "Sign Up"])
        if auth_choice == "Login":
            login()
        else:
            signup()
    else:
        st.sidebar.write(f"Logged in as: {user_id}")
        if st.sidebar.button("Logout"):
            st.session_state.pop("user_id", None)
            st.query_params = {}  # Clear query parameters on logout
            st.rerun()  # Refresh the app
# Function to extract transcript from YouTube
def extract_transcript_details(youtube_video_url):
    try:
        video_id = youtube_video_url.split("=")[1]
        transcript_text = YouTubeTranscriptApi.get_transcript(video_id)
        transcript = " ".join([i["text"] for i in transcript_text])
        return transcript
    except Exception as e:
        raise e

def generate_gemini_content(transcript_text, prompt):
    model = genai.GenerativeModel("gemini-pro")
    
    # Additional instruction to indicate when the answer is not in context
    additional = """
    You are a great reader and writer. Based on the provided transcript, answer the following question.
    If the question is not related to the provided transcript, respond with "Answer is not available in the context." Do not fabricate an answer.
    """
    
    full_prompt = additional + "\nTranscript:\n" + transcript_text + "\n\nQuestion:\n" + prompt

    # Print the prompt for debugging
    #print(full_prompt)
    
    # Generate content
    response = model.generate_content(full_prompt)
    
    # Check response for "not in context" phrase
    if "Answer is not available in the context" in response.text:
        return "Answer is not available in the context."
    
    return response.text


# Function to extract text from PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=15000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create a vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
import asyncio

def get_conversational_chain():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer and if the answer is in the context you can add your knowledge to it so that the answer is well defined but make sure that you take care of context and give well defined answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

# Function to handle user input and generate a response
def user_input(user_id, user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    save_query(user_id, user_question)  # Save the query
    st.write("Reply: ", response["output_text"])

# Function to generate an LLM response
def LLM_Response(user_id, question):
    model = genai.GenerativeModel("gemini-pro")
    chat = model.start_chat()
    response = chat.send_message(question, stream=True)
    save_query(user_id, question)
    return response
def get_model():
    return genai.GenerativeModel('gemini-1.5-flash')

def get_gemini_response(input_text, image_parts, user_prompt):
    model = get_model()  # Initialize model within the function
    response = model.generate_content([input_text, image_parts[0], user_prompt])
    return response.text

def input_image_details(uploaded_file):
    if uploaded_file is not None:
        # Read the file into bytes
        bytes_data = uploaded_file.getvalue()
        image_parts = [
            {
                "mime_type": uploaded_file.type,  # Get the mime type of the uploaded file
                "data": bytes_data
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded")
# Function to load an image
@st.cache_data
def load_image(image_file):
    img = Image.open(image_file)
    return img

# Main function to run the Streamlit app
def main():
    # Initialize session state variables
    if "counter" not in st.session_state:
        st.session_state.counter = 0

    manage_user_session()
    user_id = get_user_id_from_session()

    if user_id:
        st.title("Academic Helper CHATBOT")

        st.sidebar.title("History")
        show_history = st.sidebar.checkbox("Show History")
        if show_history:
            display_user_recent_searches(user_id)

        menu = ["Text", "Image", "YouTube", "Dataset", "PDF", "About"]
        choice = st.sidebar.selectbox("Menu", menu)

        if choice == "Text":
            st.subheader("Chat with CYPHER 💁💬")
            user_quest = st.text_input("Ask a Question", key="input_" + str(st.session_state.counter), value="")
            btn = st.button("Ask_" + str(st.session_state.counter))
            if btn and user_quest:
                st.session_state.counter += 1
                st.subheader("Response:")
                result = LLM_Response(user_id, user_quest)
                response_text = "\n".join([word.text for word in result])
                st.write(response_text)
                user_quest = st.text_input("Ask a Question", key="input_" + str(st.session_state.counter), value="")
                st.button("Ask_" + str(st.session_state.counter))

        elif choice == "Image":
            st.subheader("MultiLanguage Image Extractor 💁🖼️")
            input_prompt = st.text_input("Input Prompt:", key="input")
            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

            if uploaded_file:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image.", use_column_width=True)

            submit = st.button("Tell me about the image")

            if submit:
                if input_prompt:
                    image_data = input_image_details(uploaded_file)
                    response = get_gemini_response(input_prompt, image_data, input_prompt)
                    st.subheader("The Response is:")
                    st.write(response)
                else:
                    st.error("Please provide an input prompt.")

        elif choice == "YouTube":
            st.subheader("YouTube Video 💁💬")
            youtube_video_url = st.text_input("Enter the YouTube video link")
            prompt = st.text_area("Enter the Prompt:")

            try:
                if st.button("Get Response"):
                    transcript = extract_transcript_details(youtube_video_url)
                    response = generate_gemini_content(transcript, prompt)
                    st.write(response)
            except Exception as e:
                st.error("An error occurred: " + str(e))

        elif choice == "Dataset":
            st.subheader("Dataset ")

            # File upload
            data_file = st.file_uploader("Upload CSV", type=["csv"])

            # User question input
            user_question_dataset = st.text_input("Ask a Question about the Dataset")

            # Process button
            if st.button("Submit & Process") and data_file and user_question_dataset:
                try:
                    if data_file is not None:
                        file_details = {"Filename": data_file.name, "FileType": data_file.type, "FileSize": data_file.size}
                        st.write(file_details)

                        # Read CSV
                        df = pd.read_csv(data_file)
                        st.dataframe(df)

                        # Set pandasAI API key
                        pandasai_api_key = os.getenv("PANDASAI_API_KEY")

                        # Ensure user question is a string
                        if user_question_dataset is not None:
                            user_question_dataset = str(user_question_dataset)
                            print("User Question Dataset:", user_question_dataset)  # Debug print statement
                    
                            # Initialize Agent with the dataset
                            agent = Agent(df)

                            # Process user question
                            try:
                                response = agent.chat(user_question_dataset)
                                print("Response:", response)  # Debug print statement

                                # Check if response is not None
                                if response:
                                    # Check if response is a DataFrame, if so, display it
                                    if isinstance(response, pd.DataFrame):
                                        st.dataframe(response)
                                    else:
                                        # Otherwise, display it as a string
                                        st.write(response)
                                else:
                                    st.write("No response received. Please ask a different question.")
                            except Exception as e:
                                st.error(f"Error processing query: {e}")
                    else:
                        st.write("No file uploaded. Please upload a CSV file.")
                except Exception as e:
                    st.error(f"Error loading dataset: {e}")

        elif choice == "PDF":
            st.subheader("Query with PDF 💁📜")

            pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True, type=["pdf"])

            if st.button("Submit & Process") and pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("PDF uploaded and processed successfully!")

            user_question = st.text_input("Ask a Question from the PDF Files")

            if user_question:
                user_input(user_id, user_question)

        elif choice == "About":
            st.subheader("About Academic Helper Bot")
            st.markdown("""
                This application helps students with their queries.
                It uses various data sources to provide accurate and context-rich responses.
            """)

if __name__ == "__main__":
    main()
