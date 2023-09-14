from operator import itemgetter

from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.vectorstores import FAISS
from langchain.schema.runnable import RunnableMap
from langchain.schema import format_document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
import os
import pdfplumber
from io import StringIO
import openai

from langchain.chains import RetrievalQA


def set_llm_chat(model, temperature):
    if model == "openai/gpt-3.5-turbo":
        model = "gpt-3.5-turbo"
    if model == "openai/gpt-3.5-turbo-16k":
        model = "gpt-3.5-turbo-16k"
    if model == "openai/gpt-4":
        model = "gpt-4"
    if model == "gpt-4" or model == "gpt-3.5-turbo" or model == "gpt-3.5-turbo-16k":
        return ChatOpenAI(model=model, openai_api_base = "https://api.openai.com/v1/", openai_api_key = st.secrets["OPENAI_API_KEY"], temperature=temperature)
    else:
        headers={ "HTTP-Referer": "https://fsm-gpt-med-ed.streamlit.app", # To identify your app
          "X-Title": "GPT and Med Ed"}
        return ChatOpenAI(model = model, openai_api_base = "https://openrouter.ai/api/v1", openai_api_key = st.secrets["OPENROUTER_API_KEY"], temperature=temperature, max_tokens = 500, headers=headers)

def truncate_text(text, max_characters):
    if len(text) <= max_characters:
        return text
    else:
        truncated_text = text[:max_characters]
        return truncated_text
@st.cache_data
def load_docs(files):
    all_text = ""
    for file_path in files:
        file_extension = os.path.splitext(file_path.name)[1]
        if file_extension == ".pdf":
            pdf_reader = pdfplumber.open(file_path)
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
    # st.write(all_text)
    return all_text


@st.cache_data
def create_retriever(texts):  
    
    embeddings = OpenAIEmbeddings(model = "text-embedding-ada-002",
                                  openai_api_base = "https://api.openai.com/v1/",
                                  openai_api_key = st.secrets['OPENAI_API_KEY']
                                  )
    try:
        vectorstore = FAISS.from_texts(texts, embeddings)
    except (IndexError, ValueError) as e:
        st.error(f"Error creating vectorstore: {e}")
        return
    retriever = vectorstore.as_retriever(k=5)

    return retriever


def split_texts(text, chunk_size, overlap, split_method):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap)

    splits = text_splitter.split_text(text)
    if not splits:
        # st.error("Failed to split document")
        st.stop()

    return splits

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            # del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.write("*Please contact David Liebovitz, MD if you need an updated password for access.*")
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True
    
if "pdf_retriever" not in st.session_state:
    st.session_state.pdf_retriever = []
    
if "pdf_user_question" not in st.session_state:
    st.session_state.pdf_user_question = []

if "pdf_user_answer" not in st.session_state:
    st.session_state.pdf_user_answer = []

if "pdf_download_str" not in st.session_state:
    st.session_state.pdf_download_str = []
    
if 'model' not in st.session_state:
    st.session_state.model = "openai/gpt-3.5-turbo-16k"
    
if 'temp' not in st.session_state:
    st.session_state.temp = 0.3
    
if "last_uploaded_files" not in st.session_state:
    st.session_state["last_uploaded_files"] = []

openai.api_base = "https://openrouter.ai/api/v1"
openai.api_key = st.secrets["OPENROUTER_API_KEY"]

st.set_page_config(page_title='GPT and Med Ed', layout = 'centered', page_icon = ':stethoscope:', initial_sidebar_state = 'auto')
st.title("GPT and Medical Education")
st.write("ALPHA version 0.3")

if check_password():

    st.header("Chat with your PDFs!")
    st.info("""Embeddings, i.e., reading your file(s) and converting words to numbers, are created using an OpenAI [embedding model](https://platform.openai.com/docs/guides/embeddings/what-are-embeddings) and indexed for searching. Then,
            your selected model (e.g., gpt-3.5-turbo-16k) is used to answer your questions.""")
    st.warning("""Some PDFs are images and not formatted text. If the summary feature doesn't work, you may first need to convert your PDF
            using Adobe Acrobat. Choose: `Scan and OCR`,`Enhance scanned file` \n   Save your updates, upload and voilà, you can chat with your PDF!""")
    uploaded_files = []
    # os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

    uploaded_files = st.file_uploader("Choose your file(s)", accept_multiple_files=True)

    if uploaded_files is not None:
        documents = load_docs(uploaded_files)
        texts = split_texts(documents, chunk_size=1250,
                                    overlap=200, split_method="splitter_type")

        retriever = create_retriever(texts)

        # openai.api_base = "https://openrouter.ai/api/v1"
        # openai.api_key = st.secrets["OPENROUTER_API_KEY"]

        llm = set_llm_chat(model=st.session_state.model, temperature=st.session_state.temp)
        # llm = ChatOpenAI(model_name='gpt-3.5-turbo-16k', openai_api_base = "https://api.openai.com/v1/")

        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    else:
        st.warning("No files uploaded.")       
        st.write("Ready to answer your questions!")


    pdf_chat_option = st.radio("Select an Option", ("Summary", "Custom Question", "Generate MCQ"))
    if pdf_chat_option == "Summary":
        user_question = "Summary: Using context provided, generate a concise and comprehensive summary. Key Points: Generate a list of Key Points by using a conclusion section if present and the full context otherwise."
    if pdf_chat_option == "Custom Question":
        user_question = st.text_input("Please enter your own question about the PDF(s):")

    if st.button("Generate a Response"):
        index_context = f'Use only the reference document for knowledge. Question: {user_question}'
        pdf_answer = qa(index_context)

        # Append the user question and PDF answer to the session state lists
        st.session_state.pdf_user_question.append(user_question)
        st.session_state.pdf_user_answer.append(pdf_answer)

        # Display the PDF answer
        st.write(pdf_answer["result"])

        # Prepare the download string for the PDF questions
        pdf_download_str = f"{disclaimer}\n\nPDF Questions and Answers:\n\n"
        for i in range(len(st.session_state.pdf_user_question)):
            pdf_download_str += f"Question: {st.session_state.pdf_user_question[i]}\n"
            pdf_download_str += f"Answer: {st.session_state.pdf_user_answer[i]['result']}\n\n"

        # Display the expander section with the full thread of questions and answers
        with st.expander("Your Conversation with your PDF", expanded=False):
            for i in range(len(st.session_state.pdf_user_question)):
                st.info(f"Question: {st.session_state.pdf_user_question[i]}", icon="🧐")
                st.success(f"Answer: {st.session_state.pdf_user_answer[i]['result']}", icon="🤖")

            if pdf_download_str:
                st.download_button('Download', pdf_download_str, key='pdf_questions')
        