import streamlit as st
import validators
import requests
import time
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import YoutubeLoader, UnstructuredURLLoader
from langchain.llms import HuggingFaceHub

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="LangChain: Summarize Text From YT or Website",
    page_icon="🦜",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Custom CSS Styling
st.markdown("""
    <style>
    body {
        background-color: #f5f5dc;
        color: #000000;
        font-family: Arial, sans-serif;
    }
    .stButton > button {
        background-color: #f5f5dc;
        color: #000000;
        border: 1px solid #000000;
        padding: 10px 20px;
        border-radius: 5px;
        transition: background-color 0.3s, color 0.3s;
    }
    .stButton > button:hover {
        background-color: #8b0000;
        color: #ffffff;
    }
    .stTextInput > div > div > input {
        color: #f5f5dc;
        background-color: #000000;
        border: 1px solid #000000;
        padding: 10px;
        border-radius: 5px;
    }
    .stTextInput > div > label {
        color: #000000;
    }
    .stMarkdown, .stTitle, .stHeader, .stSubheader {
        color: #000000;
    }
    .stApp {
        background-color: #f5f5dc;
    }
    .stResult {
        color: #f5f5dc;
        background-color: #000000;
        border-radius: 5px;
        padding: 10px;
        border: 1px solid #000000;
        margin-top: 10px;
        box-sizing: border-box;
    }
    .stResult > div {
        background-color: #000000;
        color: #f5f5dc;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and Subheader
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader("Enter a YouTube or Website URL to generate a summary")

# Sidebar for Hugging Face API Key
st.sidebar.header("Configuration")
hf_api_key = st.sidebar.text_input(
    "Hugging Face API Token",
    value="",
    type="password",
    help="Enter your Hugging Face API token here."
)

# URL Input
generic_url = st.text_input("Enter URL here:")

# Function to Convert Shortened YouTube URLs
def convert_youtube_short_url(url):
    if "youtu.be" in url:
        video_id = url.split('/')[-1].split('?')[0]
        return f"https://www.youtube.com/watch?v={video_id}"
    return url

# Summarization Button
if st.button("Summarize Content"):
    # Validate Inputs
    if not hf_api_key.strip():
        st.error("Please provide a valid Hugging Face API token.")
    elif not generic_url.strip():
        st.error("Please provide a valid URL.")
    elif not validators.url(generic_url):
        st.error("The URL provided is invalid. Please enter a valid URL.")
    else:
        generic_url = convert_youtube_short_url(generic_url)
        try:
            with st.spinner("Processing..."):
                # Initialize Document Loader
                if "youtube.com" in generic_url or "youtu.be" in generic_url:
                    st.info("Loading content from YouTube...")
                    loader = YoutubeLoader.from_youtube_url(
                        generic_url,
                        add_video_info=True
                    )
                else:
                    st.info("Loading content from Website...")
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        headers={"User-Agent": "Mozilla/5.0"}
                    )
                documents = loader.load()

                if not documents:
                    st.error("Failed to retrieve content from the provided URL.")
                else:
                    # Initialize LLM
                    llm = HuggingFaceHub(
                        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
                        model_kwargs={"temperature": 0.7, "max_length": 512},
                        huggingfacehub_api_token=hf_api_key
                    )

                    # Define Prompt Template
                    prompt_template = """
                    Summarize the following content in approximately 300 words:

                    {text}
                    """
                    prompt = PromptTemplate(
                        template=prompt_template,
                        input_variables=["text"]
                    )

                    # Create Summarization Chain
                    chain = load_summarize_chain(
                        llm=llm,
                        chain_type="stuff",
                        prompt=prompt
                    )

                    # Run Chain and Generate Summary
                    summary = chain.run(documents)

                    # Display Summary
                    st.markdown(f"<div class='stResult'>{summary}</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"An error occurred: {e}")
