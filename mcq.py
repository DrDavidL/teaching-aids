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

st.set_page_config(page_title='Tools for Med Ed', layout = 'centered', page_icon = ':stethoscope:', initial_sidebar_state = 'auto')
st.title("Tools for Medical Education")
st.write("ALPHA version 0.3")

disclaimer = "placeholder"

mcq_template = """Answer the question based only on the following context:
{context}

Question: {faculty_question}
"""

mcq_generation = """Generate 3 multiple choice questions for the context provided. Follow these best practices for optimal MCQ design:

1. **Focus on a Single Learning Objective**: Each question should target a specific learning objective. Avoid "double-barreled" questions that assess multiple objectives at once.
 
2. **Ensure Clinical Relevance**: Questions should be grounded in clinical scenarios or real-world applications, especially for medical students. This ensures the assessment is relevant to their future practice.
 
3. **Avoid Ambiguity**: The wording should be clear and unambiguous. Avoid using negatives, especially double negatives, as they can be confusing.
 
4. **Use Standardized Terminology**: Stick to universally accepted medical terminology. This ensures that students are being tested on content knowledge rather than interpretation of terms.
 
5. **Avoid Tricky Questions**: The goal is to assess knowledge, not to trick students. Do not phrase negatively as in "which is NOT a correct option". Ensure that distractors (incorrect options) are plausible but clearly incorrect upon careful reading.
 
6. **Randomize Option Order**: This minimizes the chance of students guessing based on patterns.
 
7. **Avoid "All of the Above" or "None of the Above"**: These can be confusing and often don't provide clear insight into a student's understanding.
 
8 **Balance Between Recall and Application**: While some questions might test basic recall, strive to include questions that assess application, analysis, and synthesis of knowledge.
 
9. **Avoid Cultural or Gender Bias**: Ensure questions and scenarios are inclusive and don't inadvertently favor a particular group.
 
10. **Use Clear and Concise Language**: Avoid lengthy stems or vignettes unless necessary for the context. The complexity should come from the medical content, not the language.

11. **Make Plausible**: All options should be homogeneous and plausible to avoid cueing to the correct option.

12. **No Flaws**: Each item should be reviewed to identify and remove technical flaws that add irrelevant difficulty or benefit savvy test-takers."""
 


if check_password():

    st.header("Learn from your PDFs!")
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
        
    if pdf_chat_option == "Generate MCQ":
        mcq_options = st.radio("Select an Option", ("Generate 3 MCQs", "Generate MCQs on a Specific Topic"))
        
        if mcq_options == "Generate 3 MCQs":
            user_question = mcq_generation
            
        if mcq_options == "Generate MCQs on a Specific Topic":
            user_focus = st.text_input("Please enter a covered topic for the focus of your MCQ:")
            user_question = f'{mcq_generation}' + f'\n\nFocus for questions: {user_focus}'

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
        