from operator import itemgetter

from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOpenAI
# from langchain.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
# from langchain.embeddings import OpenAIEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.vectorstores import FAISS
# from langchain.vectorstores import FAISS
from langchain.schema.runnable import RunnableMap
from langchain.schema import format_document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
import os
import fitz
from io import StringIO
import openai
from openai import OpenAI
from prompts import *
import datetime
import pytz
from fpdf import FPDF
from io import BytesIO
import base64

from langchain.chains import RetrievalQA

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Chat History', 0, 1, 'C')
        
def talk_stream(model, voice, input):
    api_key = st.secrets["OPENAI_API_KEY"]
    client = OpenAI(    
    base_url="https://api.openai.com/v1",
    api_key=api_key,
)
    response = client.audio.speech.create(
    model= model,
    voice= voice,
    input= input,
    )
    response.stream_to_file("last_response.mp3")
    
def autoplay_local_audio(filepath: str):
    # Read the audio file from the local file system
    with open(filepath, 'rb') as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    md = f"""
        <audio controls autoplay="true">
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
        """
    st.markdown(
        md,
        unsafe_allow_html=True,
    )

@st.cache_data
def get_summary_from_qa(doc_content, chain_type, teaching_style, summary_template):
    with st.spinner("Generating summary for a custom chatbot"):
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type=chain_type, retriever=retriever, return_source_documents=True)
        index_context = f'Use only the reference document for knowledge. Question: {summary_template}'
        summary_for_chatbot = qa(index_context)
        return summary_for_chatbot["result"]

def set_llm_chat(model, temperature):
    if model == "openai/gpt-3.5-turbo-0125":
        model = "gpt-3.5-turbo-0125"
    if model == "openai/gpt-3.5-turbo-16k":
        model = "gpt-3.5-turbo-16k"
    if model == "openai/gpt-4-turbo-preview":
        model = "gpt-4-turbo-preview"
    if model == "gpt-4-turbo-preview" or model == "gpt-3.5-turbo-0125":
        return ChatOpenAI(model=model, openai_api_base = "https://api.openai.com/v1/", openai_api_key = st.secrets["OPENAI_API_KEY"], temperature=temperature)
    else:
        headers={ "HTTP-Referer": "https://fsm-gpt-med-ed.streamlit.app", # To identify your app
          "X-Title": "GPT and Med Ed"}
        return ChatOpenAI(model = model, openai_api_base = "https://openrouter.ai/api/v1", openai_api_key = st.secrets["OPENROUTER_API_KEY"], temperature=temperature, max_tokens = 1500, )

def truncate_text(text, max_characters):
    if len(text) <= max_characters:
        return text
    else:
        truncated_text = text[:max_characters]
        return truncated_text


@st.cache_data  # Updated decorator name from cache_data to cache
def load_docs(files):
    all_text = ""
    for file in files:
        file_extension = os.path.splitext(file.name)[1]
        if file_extension == ".pdf":
            pdf_data = file.read()  # Read the file into bytes
            pdf_reader = fitz.open("pdf", pdf_data)  # Open the PDF from bytes
            text = ""
            for page in pdf_reader:
                text += page.get_text()
            all_text += text

        elif file_extension == ".txt":
            stringio = StringIO(file.getvalue().decode("utf-8"))
            text = stringio.read()
            all_text += text
        else:
            st.warning('Please provide txt or pdf.', icon="⚠️")
    return all_text


@st.cache_resource
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
if "last_response" not in st.session_state:
    st.session_state.last_response = ""
    
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
    
if "pdf_chat_message_history" not in st.session_state:
    st.session_state.pdf_chat_message_history = []

openai.api_base = "https://openrouter.ai/api/v1"
openai.api_key = st.secrets["OPENROUTER_API_KEY"]
unplayed = False

st.set_page_config(page_title='Learn from PDFs', layout = 'centered', page_icon = ':stethoscope:', initial_sidebar_state = 'auto')
st.title("Learn from PDFs")


with st.sidebar.expander("Select a GPT Language Model", expanded=True):
    st.session_state.model = st.selectbox("Model Options", ("openai/gpt-3.5-turbo-0125", "openai/gpt-4-turbo-preview", "google/gemini-pro"), index=0)


disclaimer = """**Disclaimer:** This is a tool to assist education regarding artificial intelligence. Your use of this tool accepts the following:   
1. This tool does not generate validated medical content although attempts are made to 'ground' responses based on the user submitted PDF. \n 
2. This tool is not a real doctor. \n    
3. You will not take any medical action based on the output of this tool. \n   
"""



with st.expander('Learn from PDFs - Important Disclaimer'):
    st.write("Author: David Liebovitz, MD, Northwestern University")
    st.info(disclaimer)
    st.session_state.temp = st.slider("Select temperature (Higher values more creative but tangential and more error prone)", 0.0, 1.0, 0.5, 0.01)
    st.warning("""Some PDFs are images and not formatted text. If an appropriate outline fails to appear, you may first need to convert your PDF
        using Adobe Acrobat. Choose: `Scan and OCR`,`Enhance scanned file` \n   Alternatively, sometimes PDFs are created with 
        unusual fonts or LaTeX symbols. Export the file to Word, re-save as a PDF and try again. Save your updates, upload and voilà, you can chat with your PDF! """)
    st.write("Last updated 2/13/24")


 


if check_password():

    # st.header("Analyze your PDFs!")


    st.info("""This system uses Retrieval Augmented Generation [(RAG)](https://arxiv.org/abs/2005.11401) for interactions with your PDF. The embeddings, i.e., transformations of the words in your PDF into vectors, are created using an OpenAI [embedding model](https://platform.openai.com/docs/guides/embeddings/what-are-embeddings) 
        and then stored in a [FAISS](https://github.com/facebookresearch/faiss) similarity search vector database. Your selected model (e.g., gpt-3.5-turbo-0125) is then used to formulate a final response to your questions.""")
    
    uploaded_files = []
    # os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

    uploaded_files = st.file_uploader("Choose your file(s). Currently uses only your last uploaded file. if your PDF is readable by the tool, an outline will appear to the left.", accept_multiple_files=True)

    if uploaded_files is not None:
        documents = load_docs(uploaded_files)
        texts = split_texts(documents, chunk_size=1250,
                                    overlap=200, split_method="splitter_type")

        retriever = create_retriever(texts)

        # openai.api_base = "https://openrouter.ai/api/v1"
        # openai.api_key = st.secrets["OPENROUTER_API_KEY"]

        llm = set_llm_chat(model=st.session_state.model, temperature=st.session_state.temp)
        # llm = ChatOpenAI(model_name='gpt-3.5-turbo-16k', openai_api_base = "https://api.openai.com/v1/")

        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True,)
        
        # Set the context for the subsequent chatbot conversation
        prepare_summary = key_points_list_for_chatbot.format(context = "{context}")
             
        with st.spinner("Generating summary for a custom chatbot"):
            summary_for_chatbot = get_summary_from_qa(documents, "stuff", "", prepare_summary)
            
        with st.sidebar.expander("Topics available to your Chatbot", expanded=True):
            st.info("Topics Identified in the PDF")
            st.write(summary_for_chatbot)

    else:
        st.warning("No files uploaded.")       
        st.write("Ready to answer your questions!")

    col1, col2 = st.columns(2)
    with col1:
        pdf_chat_option = st.selectbox("Select an Option", ("Ask Questions about your PDF", "Generate MCQs from your PDF", "Summarize your PDF", "Appraise a Clinical Trial PDF"))
    
    with col2:
        st.session_state.audio_off = st.checkbox("Turn off voice generation", value=False) 
    
    if pdf_chat_option == "Appraise a Clinical Trial PDF":
        st.write('Note GPT4 is much better; may take a few minutes to run.')
        word_count = st.slider("~Word Count for the Summary. Most helpful for very long articles", 100, 1000, 250)
        user_question = clinical_trial_template
        user_question = user_question.format(word_count=word_count, context = "{context}")
    
    if pdf_chat_option == "Summarize your PDF":        
        # user_question = "Summary: Using context provided, generate a concise and comprehensive summary. Key Points: Generate a list of Key Points by using a conclusion section if present and the full context otherwise."
        with col2:
            summary_method= st.radio("Select a Summary Method", ("Standard Summary", "Chain of Density"))
        word_count = st.slider("Approximate Word Count for the Summary. Most helpful for very long articles", 100, 1000, 250)
        if summary_method == "Chain of Density":
            st.write("Generated with [Chain of Density](https://arxiv.org/abs/2309.04269) methodology.")
            user_question = chain_of_density_summary_template
            user_question = user_question.format(word_count=word_count, context = "{context}")
        if summary_method == "Standard Summary":
            user_question = key_points_summary_template
            user_question = user_question.format(word_count=word_count, context = "{context}")
        # user_question = "Summary: Using context provided, generate a concise and comprehensive summary. Key Points: Generate a list of Key Points by using a conclusion section if present and the full context otherwise."

        
    if pdf_chat_option == "Generate MCQs from your PDF":
        num_mcq = st.slider("Number of MCQs", 1, 10, 3)
        with col2: 
            mcq_options = st.radio("Select a Sub_Option", ("Generate MCQs", "Generate MCQs on a Specific Topic"))
        
        if mcq_options == "Generate MCQs":
            user_question = mcq_generation_template
            user_question = user_question.format(num_mcq=num_mcq, context = "{context}")
            
        if mcq_options == "Generate MCQs on a Specific Topic":
            user_focus = st.text_input("Please enter a covered topic for the focus of your MCQ:")
            user_question = f'Topic for question generation: {user_focus}' + f'\n\n {mcq_generation_template}'
            user_question = user_question.format(num_mcq=num_mcq, context = "{context}")

    if pdf_chat_option == "Ask Questions about your PDF":
        

        # Capture user input. If the user enters a question, proceed with generating a response.
        if follow_up_question := st.chat_input("Please ask questions about your PDF here!"):
            # Append the user's question to the chat history.
            
            

            st.session_state.pdf_chat_message_history.append({"role": "user", "content": follow_up_question})
            # st.session_state.message_history.append({"role": "user", "content": follow_up_question})

            # Display the user's question in the chat interface.
            with st.chat_message("user"):
                st.markdown(follow_up_question)

            # Generate and display the assistant's response.
            with st.chat_message("assistant"):
                
                index_context = f'Use only the reference document for knowledge. User question: {follow_up_question}'
            
                pdf_answer = qa(index_context)
                # Create a chat completion request to the OpenAI API, passing in the model and the conversation history.
                response = st.write(pdf_answer["result"])
                st.session_state.last_response = pdf_answer["result"]
                unplayed = True
            # Append the assistant's response to the chat history.
            st.session_state.pdf_chat_message_history.append({"role": "assistant", "content": pdf_answer["result"]})


        if st.session_state.get('pdf_chat_message_history'):  # Ensure message_history is in session_state
            make_pdf = st.checkbox("Create a PDF of your Q/A for download?")
            if make_pdf:
                with st.spinner('Generating PDF...'):
                    try:
                        pdf = PDF()
                        pdf.add_page()
                        pdf.set_font("Arial", size=12)
                        # Format the datetime for display
                        central_time = pytz.timezone('America/Chicago')
                        specific_datetime = datetime.datetime.now()
                        specific_datetime_central = central_time.localize(specific_datetime)
                        datetime_str = specific_datetime_central.strftime('%A %b %d, %Y at %I:%M %p')
                        
                        # Add the formatted datetime to the top of the PDF content
                        pdf.cell(0, 10, datetime_str, 0, 1, 'C')  # Adjust alignment as needed
                        
                        for message in st.session_state.pdf_chat_message_history:
                            if message["role"] != "system":
                                text = f"{message['role']}: {message['content']}"
                                pdf.multi_cell(0, 10, text)
                                pdf.cell(0, 10, "*" * 20, 0, 1)

                        # Generate PDF content as a byte string
                        pdf_content = pdf.output(dest='S').encode('latin1')  # 'S' returns the document as a string, then encode to bytes

                        st.download_button(label="Download Chat History as PDF",
                                        data=pdf_content,
                                        file_name="chat_history.pdf",
                                        mime="application/pdf")
                        st.success('PDF successfully generated!')

                    except Exception as e:
                        st.error(f'An error occurred: {e}')

        
        
        
        
        
        
    if pdf_chat_option != "Ask Questions about your PDF":
        if st.button("Generate a Response"):
            index_context = f'Use only the reference document for knowledge. Question: {user_question}'
            
            pdf_answer = qa(index_context)

            # Append the user question and PDF answer to the session state lists
            st.session_state.pdf_user_question.append(user_question)
            st.session_state.pdf_user_answer.append(pdf_answer)

            # Display the PDF answer
            st.write(pdf_answer["result"])
            st.session_state.last_response = pdf_answer["result"]
            unplayed = True
            

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
        
            
    if st.session_state.audio_off == False and unplayed == True:
        if st.session_state.last_response != "":
            
            #     patient_section = extract_patient_response(st.session_state.last_response_interview)
            # st.write(patient_section)
             
                # Define the data
            # path_audio = play_audio_eleven(st.session_state.last_response_interview, voice=voice)
            talk_stream("tts-1", "shimmer", st.session_state.last_response)
            
            # data = {
            #     "text": st.session_state.last_response_interview,
            #     "voice": voice,
            # }

            # Send the POST request
            # response_from_audio = requests.post(audio_url, headers=headers, data=json.dumps(data))
            # st.sidebar.write(response_from_audio.text)
            # st.write(f'Audio full: {response_from_audio.text}')
            # st.write(f'Audio url: {response_from_audio.json()}')
            # Print the response
            # link_to_audio = extract_url(response_from_audio.text)
            # st.write(path_audio)
            autoplay_local_audio("last_response.mp3")
            with st.expander("Audio Transcript", expanded=True):
                st.write(st.session_state.last_response)
            unplayed = False
        else:
            pass