import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from langchain_huggingface import HuggingFaceEndpoint

# Custom CSS
st.markdown("""
    <style>
    body {
        background-color: beige;
        color: black;
        font-family: Arial, sans-serif;
    }
    .stButton > button {
        background-color: beige;
        color: black;
        border: 1px solid black;
        padding: 10px 20px;
        border-radius: 5px;
        transition: background-color 0.3s, color 0.3s;
    }
    .stButton > button:hover {
        background-color: darkred;
        color: black;
    }
    .stTextInput > div > div > input {
        color: beige;
        background-color: black;
        border: 1px solid black;
        padding: 10px;
        border-radius: 5px;
    }
    .stTextInput > div > label {
        color: black;
    }
    .stMarkdown, .stTitle, .stHeader, .stSubheader {
        color: black;
    }
    .stApp {
        background-color: beige;
    }
    .stResult {
        color: beige;
        background-color: black;
        border-radius: 5px;
        padding: 10px;
        border: 1px solid black;
        margin-top: 10px;
        width: 200%;
        box-sizing: border-box;
    }
    .stResult > div {
        background-color: black;
        color: beige;
    }
    .stText, .stTextInput, .stMarkdown {
        color: black;
    }
    .stTextInput > div > div > input {
        color: beige;
        background-color: black;
        border: 1px solid black;
    }
    .stMarkdown {
        color: black;
    }
    .url-history {
        display: flex;
        flex-wrap: wrap;
        gap: 20px;
        margin-top: 20px;
    }
    .url-entry {
        position: relative;
        width: 45%;
        background-color: black;
        color: beige;
        border-radius: 5px;
        overflow: hidden;
        transition: background-color 0.3s, color 0.3s;
    }
    .url-entry:hover .url-summary {
        display: block;
    }
    .url-summary {
        display: none;
        padding: 10px;
        color: beige;
        background-color: black;
        border-top: 1px solid beige;
    }
    .url-link {
        padding: 10px;
        background-color: black;
        color: beige;
        text-align: center;
        font-weight: bold;
        border-bottom: 1px solid beige;
        transition: background-color 0.3s, color 0.3s;
    }
    .url-entry:hover .url-link {
        background-color: darkred;
        color: black;
    }
    </style>
    """, unsafe_allow_html=True)

# Streamlit App Configuration
st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader('Summarize URL')

# Get the Hugging Face API Key and URL to be summarized
with st.sidebar:
    hf_api_key = st.text_input("Huggingface API Token", value="", type="password")

generic_url = st.text_input("URL", label_visibility="collapsed")

# Convert shortened YouTube URL if needed
def convert_youtube_short_url(url):
    if "youtu.be" in url:
        video_id = url.split('/')[-1].split('?')[0]
        return f"https://www.youtube.com/watch?v={video_id}"
    return url

generic_url = convert_youtube_short_url(generic_url)

# LLM Model using Hugging Face API
repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
llm = HuggingFaceEndpoint(repo_id=repo_id, max_length=150, temperature=0.7, token=hf_api_key)

prompt_template = """
Provide a summary of the following content in 300 words:
Content: {text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

# Button to Summarize the Content
if st.button("Summarize the Content from YT or Website"):
    # Validate all the inputs
    if not hf_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL. It can be a YT video URL or website URL")
    else:
        try:
            with st.spinner("Waiting..."):
                # Load the website or YouTube video data
                if "youtube.com" in generic_url:
                    loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=True)
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"}
                    )
                docs = loader.load()

                # Chain for Summarization
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                output_summary = chain.run(docs)

                st.success(output_summary)
        except Exception as e:
            st.exception(f"Exception: {e}")
