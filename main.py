    # Import necessary libraries
import os  # Used for accessing environment variables like our secret OpenAI API key.
from openai import OpenAI  # Import the OpenAI library to interact with the API
import streamlit as st  # Streamlit library for creating web apps
from prompts import *  # Import predefined prompts

from fpdf import FPDF
from io import BytesIO
def check_password2():
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

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Chat History', 0, 1, 'C')


st.title('Medical Educator Chat Playground')
st.info('This is a simple playground to investigate LLM use. Features include enhanced chats, prompt engineering, MCQ generation, and more.')


if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "improved_question" not in st.session_state:
    st.session_state["improved_question"] = ""
    
if "messsages" not in st.session_state:
    st.session_state["messages"] = []

prompt_engineering_strategies = [
    "Self Discovery Prompting",
    "Imperfect Prompting",
    "Persistent Context and Custom Instructions Prompting",
    "Multi-Persona Prompting",
    "Chain-of-Thought (CoT) Prompting",
    "Retrieval-Augmented Generation (RAG) Prompting",
    "Chain-of-Thought Factored Decomposition Prompting",
    "Skeleton-of-Thought (SoT) Prompting",
    "Show-Me Versus Tell-Me Prompting",
    "Mega-Personas Prompting",
    "Certainty and Uncertainty Prompting",
    "Vagueness Prompting",
    "Catalogs or Frameworks for Prompting",
    "Flipped Interaction Prompting",
    "Self-Reflection Prompting",
    "Add-On Prompting",
    "Conversational Prompting",
    "Prompt-to-Code Prompting",
    "Target-Your-Response (TAYOR) Prompting",
    "Macros and End-Goal Prompting",
    "Tree-of-Thoughts (ToT) Prompting",
    "Trust Layers for Prompting",
    "Directional Stimulus Prompting (DSP)",
    "Privacy Invasive Prompting",
    "Illicit or Disallowed Prompting",
    "Chain-of-Density (CoD) Prompting",
    "Take a Deep Breath Prompting",
    "Chain-of-Verification (CoV) Prompting",
    "Beat the Reverse Curse Prompting",
    "Overcoming Dumbing Down Prompting",
    "DeepFakes to TrueFakes Prompting",
    "Disinformation Detection and Removal Prompting",
    "Emotionally Expressed Prompting"
]
method_mapping = {
    "Imperfect Prompting": problem_solving_Imperfect_Prompting,
    "Persistent Context and Custom Instructions Prompting": problem_solving_Persistent_Context_and_Custom_Instructions_Prompting,
    "Multi-Persona Prompting": problem_solving_Multi_Persona_Prompting,
    "Chain-of-Thought (CoT) Prompting": problem_solving_Chain_of_Thought_CoT_Prompting,
    "Retrieval-Augmented Generation (RAG) Prompting": problem_solving_Retrieval_Augmented_Generation_RAG_Prompting,
    "Chain-of-Thought Factored Decomposition Prompting": problem_solving_Chain_of_Thought_Factored_Decomposition_Prompting,
    "Skeleton-of-Thought (SoT) Prompting": problem_solving_Skeleton_of_Thought_SoT_Prompting,
    "Show-Me Versus Tell-Me Prompting": problem_solving_Show_Me_Versus_Tell_Me_Prompting,
    "Mega-Personas Prompting": problem_solving_Mega_Personas_Prompting,
    "Certainty and Uncertainty Prompting": problem_solving_Certainty_and_Uncertainty_Prompting,
    "Vagueness Prompting": problem_solving_Vagueness_Prompting,
    "Catalogs or Frameworks for Prompting": problem_solving_Catalogs_or_Frameworks_for_Prompting,
    "Flipped Interaction Prompting": problem_solving_Flipped_Interaction_Prompting,
    "Self-Reflection Prompting": problem_solving_Self_Reflection_Prompting,
    "Add-On Prompting": problem_solving_Add_On_Prompting,
    "Conversational Prompting": problem_solving_Conversational_Prompting,
    "Prompt-to-Code Prompting": problem_solving_Prompt_to_Code_Prompting,
    "Target-Your-Response (TAYOR) Prompting": problem_solving_Target_Your_Response_TAYOR_Prompting,
    "Macros and End-Goal Prompting": problem_solving_Macros_and_End_Goal_Prompting,
    "Tree-of-Thoughts (ToT) Prompting": problem_solving_Tree_of_Thoughts_ToT_Prompting,
    "Trust Layers for Prompting": problem_solving_Trust_Layers_for_Prompting,
    "Directional Stimulus Prompting (DSP)": problem_solving_Directional_Stimulus_Prompting_DSP,
    "Privacy Invasive Prompting": problem_solving_Privacy_Invasive_Prompting,
    "Illicit or Disallowed Prompting": problem_solving_Illicit_or_Disallowed_Prompting,
    "Chain-of-Density (CoD) Prompting": problem_solving_Chain_of_Density_CoD_Prompting,
    "Take a Deep Breath Prompting": problem_solving_Take_a_Deep_Breath_Prompting,
    "Chain-of-Verification (CoV) Prompting": problem_solving_Chain_of_Verification_CoV_Prompting,
    "Beat the Reverse Curse Prompting": problem_solving_Beat_the_Reverse_Curse_Prompting,
    "Overcoming Dumbing Down Prompting": problem_solving_Overcoming_Dumbing_Down_Prompting,
    "DeepFakes to TrueFakes Prompting": problem_solving_DeepFakes_to_TrueFakes_Prompting,
    "Disinformation Detection and Removal Prompting": problem_solving_Disinformation_Detection_and_Removal_Prompting,
    "Emotionally Expressed Prompting": problem_solving_Emotionally_Expressed_Prompting,
    "Self Discovery Prompting": problem_solving_Self_Discover_Prompting,
}

if check_password2():
    with st.sidebar:
        with st.expander("Set Model Options"):
            # Show model options
            st.session_state.openai_model = st.selectbox("Select a model", ["gpt-3.5-turbo-0125", "gpt-4-turbo-preview"])
            
        with st.expander("Prompt Engineering Strategies"):
            choose_system_prompt = st.selectbox("Choose a system prompt", ["Medical Educator System", "General Expert", "Your own!"])
            if choose_system_prompt == "Your own!":
                system_prompt = st.text_area("Enter your own prompt")

            elif choose_system_prompt == "Medical Educator System":
                system_prompt = medical_educator_system_prompt
                
            elif choose_system_prompt == "General Expert":
                system_prompt = system_prompt_Generic_Expert_Prompting
                    
            problem_solving_strategies = st.multiselect(
                "Select one or more problem solving strategies:",
                options=prompt_engineering_strategies,
                default=None
            )
            strategy_string = "None specified"
            for strategy in problem_solving_strategies:
                strategy_string = method_mapping.get(strategy, "None specified")
                
        if st.checkbox("Show system prompt"):
            st.write(system_prompt)
        
        if st.checkbox("Show problem solving strategy"):

            st.write(f"Selected: {strategy_string}")

    user_prompt = st.text_input("Enter your question:")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    if st.button("Make my question better!"):


        improved_question = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": "system", "content": system_prompt_improve_question},
                {"role": "user", "content": user_prompt}
            ],
            stream=False,
        )
        st.session_state["improved_question"] = improved_question.choices[0].message.content
    # Display the response from the API.
    if st.session_state.improved_question:
        st.text_area("Improved Question", st.session_state.improved_question, height=150, key="improved_question_text_area")
    use_original = st.checkbox("Send original question (otherwise will send improved question)")
    send_question = st.button("Submit Question to start chat (clears chat history")

    # Display previous chat messages. This loop goes through all messages and displays them, skipping system messages.
    if st.session_state.messages is not []:
        for message in st.session_state.messages:
            if message["role"] != "system":
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])    

    if send_question:
        st.session_state.messages =[]
        if use_original:
            first_prompt = user_prompt
        else:
            first_prompt = st.session_state.improved_question
            # Append the user's question to the chat history.
        if strategy_string:
            system_prompt = system_prompt + "Apply problem solving strategy: " + strategy_string
        st.session_state.messages.append({"role": "system", "content": system_prompt})
        st.session_state.messages.append({"role": "user", "content": first_prompt})
        st.session_state.message_history.append({"role": "user", "content": first_prompt})

        # Display the user's question in the chat interface.
        with st.chat_message("user"):
            st.markdown(first_prompt)

        # Generate and display the assistant's response.
        with st.chat_message("assistant"):
            # Create a chat completion request to the OpenAI API, passing in the model and the conversation history.
            stream = client.chat.completions.create(
                model=st.session_state["openai_model"],
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=True,
            )
            # Display the response from the API.
            response = st.write_stream(stream)
        # Append the assistant's response to the chat history.
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.message_history.append({"role": "assistant", "content": response})


        
        





    # Retrieve the OpenAI API key from environment variables for security reasons.
    openai_api_key = os.getenv("OPENAI_API_KEY")

    # Create an OpenAI client instance. This is necessary for sending requests to the OpenAI API.
    client = OpenAI(api_key=openai_api_key)

    # Check if the model selection has been made; if not, set a default model.
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-3.5-turbo-0125"

    # Initialize the chat history. Since GPT doesn't remember previous interactions, we need to manage the conversation history ourselves.
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Add a system message to set the context for the GPT model.
        st.session_state.messages.append({"role": "system", "content": medical_educator_system_prompt})
        




    # Capture user input. If the user enters a question, proceed with generating a response.
    if prompt := st.chat_input("Please ask follow-up questions here!"):
        # Append the user's question to the chat history.
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.message_history.append({"role": "user", "content": prompt})

        # Display the user's question in the chat interface.
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display the assistant's response.
        with st.chat_message("assistant"):
            # Create a chat completion request to the OpenAI API, passing in the model and the conversation history.
            stream = client.chat.completions.create(
                model=st.session_state["openai_model"],
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=True,
            )
            # Display the response from the API.
            response = st.write_stream(stream)
        # Append the assistant's response to the chat history.
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.message_history.append({"role": "assistant", "content": response})

    with st.expander("Full Chat History and Printing"):
        # Display previous chat messages. This loop goes through all messages and displays them, skipping system messages.
        
        for message in st.session_state.message_history:
            if message["role"] != "system":
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

    if st.session_state.get('message_history'):  # Ensure message_history is in session_state
        make_pdf = st.checkbox("Create a PDF for download?")
        if make_pdf:
            with st.spinner('Generating PDF...'):
                try:
                    pdf = PDF()
                    pdf.add_page()
                    pdf.set_font("Arial", size=12)
                    for message in st.session_state.message_history:
                        if message["role"] != "system":
                            text = f"{message['role']}: {message['content']}"
                            pdf.multi_cell(0, 10, text)

                    # Generate PDF content as a byte string
                    pdf_content = pdf.output(dest='S').encode('latin1')  # 'S' returns the document as a string, then encode to bytes

                    st.download_button(label="Download Chat History as PDF",
                                    data=pdf_content,
                                    file_name="chat_history.pdf",
                                    mime="application/pdf")
                    st.success('PDF successfully generated!')

                except Exception as e:
                    st.error(f'An error occurred: {e}')