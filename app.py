import streamlit as st
import validators
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from langchain.chains.summarize import load_summarize_chain

# Custom CSS for flip cards and styling
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
        color: beige;
    }
    .stTextInput > div > div > input {
        color: black;
        background-color: beige;
        border: 1px solid black;
        padding: 10px;
        border-radius: 5px;
    }
    .stTextInput > div > label {
        color: black;
    }
    .flip-card {
        background-color: transparent;
        width: 300px;
        height: 200px;
        perspective: 1000px;
    }
    .flip-card-inner {
        position: relative;
        width: 100%;
        height: 100%;
        text-align: center;
        transition: transform 0.6s;
        transform-style: preserve-3d;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
    }
    .flip-card:hover .flip-card-inner {
        transform: rotateY(180deg);
    }
    .flip-card-front, .flip-card-back {
        position: absolute;
        width: 100%;
        height: 100%;
        backface-visibility: hidden;
        border-radius: 5px;
    }
    .flip-card-front {
        background-color: beige;
        color: black;
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .flip-card-back {
        background-color: black;
        color: beige;
        transform: rotateY(180deg);
        padding: 10px;
        box-sizing: border-box;
    }
    .url-history {
        display: flex;
        flex-wrap: wrap;
        gap: 20px;
        margin-top: 20px;
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
            st.experimental_set_query_params(auth="true")
            st.experimental_rerun()
        else:
            st.error("Please provide the Hugging Face API key to proceed.")
else:
    # Main functionality page after API key is provided
    st.title("🦜 LangChain: Summarize Text From YT or Website")
    st.subheader('Summarize URL')

    # Input field for URL
    generic_url = st.text_input("URL", label_visibility="collapsed")

    def convert_youtube_short_url(url):
        if "youtu.be" in url:
            video_id = url.split('/')[-1].split('?')[0]
            return f"https://www.youtube.com/watch?v={video_id}"
        return url

    generic_url = convert_youtube_short_url(generic_url)

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
                    with st.spinner("Waiting..."):
                        # Add the current URL to history
                        st.session_state.url_history.append({"url": generic_url, "summary": ""})

                        # Loading the website or YT video data
                        if "youtube.com" in generic_url:
                            loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=True)
                        else:
                            loader = UnstructuredURLLoader(
                                urls=[generic_url],
                                ssl_verify=False,
                                headers={"User-Agent": "your-bot 0.1"}
                            )
                        docs = loader.load()

                        # Chain for Summarization
                        repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
                        llm = HuggingFaceEndpoint(repo_id=repo_id, max_length=150, temperature=0.7, token=st.session_state.hf_api_key)

                        prompt_template = """
                        Provide a summary of the following content in 300 words:
                        Content: {text}
                        """
                        prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
                        chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)

                        output_summary = chain.run(docs)

                        # Update session state with the summary
                        st.session_state.url_history[-1]["summary"] = output_summary
                        st.success(output_summary)
                except Exception as e:
                    st.exception(f"Exception: {e}")

    with col2:
        if st.button("Clear"):
            st.session_state.url_history = []
            st.experimental_rerun()

    # Display URL History with summaries in flip cards
    if st.session_state.url_history:
        st.write("### URL History")
        url_history_container = st.container()
        with url_history_container:
            st.markdown('<div class="url-history">', unsafe_allow_html=True)
            for item in st.session_state.url_history:
                st.markdown(f"""
                    <div class="flip-card">
                        <div class="flip-card-inner">
                            <div class="flip-card-front">
                                <a href="{item['url']}" target="_blank">{item['url']}</a>
                            </div>
                            <div class="flip-card-back">
                                {item['summary']}
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
