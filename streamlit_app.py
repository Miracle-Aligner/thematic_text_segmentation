import json
import os
from dotenv import load_dotenv

import streamlit as st

import nltk
from nltk.corpus import reuters

nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("reuters")
nltk.download("punkt")
nltk.download("punkt_tab")

from text_processor import ThematicTextProcessor

load_dotenv()


def main():
    st.title("Thematic Text Processing with BERTopic")

    # Sidebar settings
    use_openai = st.sidebar.checkbox("Use OpenAI for topic representation")
    api_key = os.getenv("OPENAI_API_KEY", "")
    if use_openai:
        if not api_key:
            api_key = st.sidebar.text_input(
                "Enter your OpenAI API Key", type="password"
            )
        if not api_key:
            st.warning(
                "Please enter your OpenAI API Key to use OpenAI representations."
            )

    # Choose input method
    data_source = st.sidebar.radio(
        "Select Data Source", ("Use Reuters Dataset", "Provide Custom Text")
    )

    if data_source == "Provide Custom Text":
        user_text = st.text_area("Enter your text here")
        uploaded_file = st.file_uploader("Or upload a text file", type=["txt"])
        if uploaded_file is not None:
            user_text = uploaded_file.read().decode("utf-8")
    else:
        # Use Reuters dataset
        user_text = " ".join(reuters.raw(fileid) for fileid in reuters.fileids()[:50])

    if st.button("Create Topics"):
        if not user_text.strip():
            st.error("Please provide some text to process.")
            return

        # Initialize processor
        processor = ThematicTextProcessor(api_key=api_key, use_openai=use_openai)
        with st.spinner("Processing text and creating topics..."):
            result_list = processor.process_text(user_text)

        st.success("Topic creation complete!")

        json_data = json.dumps(result_list, indent=4)
        st.download_button(
            label="Download Topics as JSON",
            data=json_data,
            file_name="topics.json",
            mime="application/json",
        )

        st.json(result_list)


if __name__ == "__main__":
    main()
