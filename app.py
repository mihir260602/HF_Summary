import streamlit as st
import validators
import time
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from langchain.chains import load_summarize_chain
import requests

# Custom CSS
st.markdown("""
    <style>
    body {
        background-color: black;
        color: beige;
        font-family: Arial, sans-serif;
    }
    .stButton > button {
        background-color: black;
        color: beige;
        border: 1px solid beige;
        padding: 10px 20px;
        border-radius: 5px;
        transition: background-color 0.3s, color 0.3s;
    }
    .stButton > button:hover {
        background-color: darkred;
        color: beige;
    }
    .stTextInput > div > div > input {
        color: black;
        background-color: beige;
        border: 1px solid beige;
        padding: 10px;
        border-radius: 5px;
    }
    .stTextInput > div > label {
        color: beige;
    }
    .stMarkdown, .stTitle, .stHeader, .stSubheader {
        color: beige;
    }
    .stApp {
        background-color: black;
    }
    .stResult {
        color: black;
        background-color: beige;
        border-radius: 5px;
        padding: 10px;
        border: 1px solid beige;
        margin-top: 10px;
        width: 150%;
        box-sizing: border-box;
    }
    .stResult > div {
        background-color: beige;
        width: 150%;
        color: black;
    }
    .stText, .stTextInput, .stMarkdown {
        color: beige;
    }
    .stTextInput > div > div > input {
        color: black;
        background-color: beige;
        border: 1px solid beige;
    }
    .stMarkdown {
        color: beige;
    }
    .url-history {
        display: flex;
        flex-wrap: wrap;
        gap: 20px;
        margin-top: 20px;
    }
    .url-entry {
        position: relative;
        width: 150%; /* Updated width */
        background-color: beige;
        color: black;
        border-radius: 5px;
        overflow: hidden;
        transition: background-color 0.3s, color 0.3s;
        box-sizing: border-box;
    }
    .url-entry:hover .url-summary {
        display: block;
    }
    .url-summary {
        display: none;
        padding: 10px;
        color: black;
        background-color: beige;
        border-top: 1px solid black;
    }
    .url-link {
        padding: 10px;
        background-color: beige;
        color: black;
        text-align: center;
        font-weight: bold;
        border-bottom: 1px solid black;
        transition: background-color 0.3s, color 0.3s;
    }
    .url-entry:hover .url-link {
        background-color: darkred;
        color: beige;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state for API key, URL history, and summaries
if "hf_api_key" not in st.session_state:
    st.session_state.hf_api_key = ""
if "url_history" not in st.session_state:
    st.session_state.url_history = []

# Page to enter Hugging Face API Key
if not st.session_state.hf_api_key:
    st.title("Enter Hugging Face API Key")
    st.session_state.hf_api_key = st.text_input("Hugging Face API Token", value="", type="password")

    if st.button("Submit"):
        if st.session_state.hf_api_key.strip():
            st.experimental_rerun()
        else:
            st.error("Please provide the Hugging Face API key to proceed.")
else:
    # Main functionality page after API key is provided
    st.title("🦜 LangChain: Summarize Text From YT or Website")
    st.subheader('Summarize URL')

    # Input field for URL
    generic_url = st.text_input("URL", label_visibility="collapsed")

    # Summarize and Clear buttons
    col1, col2 = st.columns([2, 1])
    with col1:
        if st.button("Summarize the Content from YT or Website"):
            # Validate all the inputs
            if not generic_url.strip():
                st.error("Please provide the URL to get started.")
            elif not validators.url(generic_url):
                st.error("Please enter a valid URL. It can be a YT video URL or website URL.")
            else:
                try:
                    st.write(f"Processing URL: {generic_url}")
                    with st.spinner("Waiting..."):
                        # Add the current URL to history
                        st.session_state.url_history.append({"url": generic_url, "summary": ""})

                        # Convert shortened YouTube URL if needed
                        def convert_youtube_short_url(url):
                            if "youtu.be" in url:
                                video_id = url.split('/')[-1].split('?')[0]
                                return f"https://www.youtube.com/watch?v={video_id}"
                            return url

                        generic_url = convert_youtube_short_url(generic_url)

                        # Custom function to fetch content with a user-agent header
                        def fetch_content(url):
                            headers = {
                                'User-Agent': 'your-bot 0.1'
                            }
                            response = requests.get(url, headers=headers)
                            response.raise_for_status()  # Raise HTTPError for bad responses
                            return response.text

                        # Loading the website or YT video data
                        if "youtube.com" in generic_url:
                            st.write("Detected YouTube URL. Attempting to load video...")
                            loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=True)
                        else:
                            st.write("Detected website URL. Attempting to load...")
                            content = fetch_content(generic_url)
                            loader = UnstructuredURLLoader(
                                urls=[generic_url],
                                ssl_verify=False,
                                headers={"User-Agent": "your-bot 0.1"}
                            )
                            # Mock loading since we don't have actual loader support for raw HTML
                            docs = [{'page_content': content}]
                        
                        docs = loader.load()

                        # Check what was loaded
                        if not docs or not docs[0].page_content.strip():
                            st.error("Unable to retrieve content from the provided URL.")
                        else:
                            st.write(f"Retrieved {len(docs)} documents.")
                            # Chain for Summarization
                            repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
                            llm = HuggingFaceEndpoint(repo_id=repo_id, max_length=150, temperature=0.7, token=st.session_state.hf_api_key)

                            prompt_template = """
                            Provide a summary of the following content in 300 words:
                            Content: {text}
                            """
                            prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
                            chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)

                            # Retry logic for handling rate limits
                            retry_attempts = 5  # Increased number of retries
                            delay = 10  # Starting delay in seconds

                            for attempt in range(retry_attempts):
                                try:
                                    output_summary = chain.run(docs)
                                    # Update session state with the summary
                                    st.session_state.url_history[-1]["summary"] = output_summary
                                    st.markdown(f"<div class='stResult'>{output_summary}</div>", unsafe_allow_html=True)
                                    break
                                except Exception as e:
                                    if "429 Client Error: Too Many Requests" in str(e):
                                        if attempt < retry_attempts - 1:
                                            st.write(f"Rate limit exceeded. Retrying in {delay} seconds...")
                                            time.sleep(delay)
                                            delay *= 2  # Exponential backoff
                                        else:
                                            st.error("Rate limit exceeded. Please try again later or upgrade your API plan.")
                                    else:
                                        st.exception(f"Exception: {e}")
                                        break
                except Exception as e:
                    st.exception(f"Exception: {e}")

    with col2:
        if st.button("Clear"):
            st.session_state.url_history = []
            st.experimental_rerun()

    # Display URL History with summaries and URLs side by side
    if st.session_state.url_history:
        st.write("### URL History")
        url_history_container = st.container()
        with url_history_container:
            st.markdown('<div class="url-history">', unsafe_allow_html=True)
            for item in st.session_state.url_history:
                st.markdown(f"""
                    <div class="url-entry">
                        <div class="url-link">
                            <a href="{item['url']}" target="_blank">{item['url']}</a>
                        </div>
                        <div class="url-summary">
                            {item['summary']}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
