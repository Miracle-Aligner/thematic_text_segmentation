import json
import os
from dotenv import load_dotenv

import openai
from bertopic import BERTopic
from bertopic.representation import OpenAI
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

import nltk
from nltk.corpus import reuters
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("reuters")
nltk.download("punkt")
nltk.download("punkt_tab")


class ThematicTextProcessor:
    def __init__(self, api_key=None, use_openai=False, openai_model="gpt-4o-mini"):
        self.api_key = api_key
        self.use_openai = use_openai
        self.openai_model = openai_model
        self.topic_model = None
        self.mapped_chunks = {}
        self.processed_chunks = []

        if self.use_openai and self.api_key:
            openai.api_key = self.api_key
            self.representation_model = OpenAI(
                openai.OpenAI(api_key=self.api_key),
                model=self.openai_model,
                chat=True,
            )
        else:
            self.representation_model = None

    def chunk_text_by_sentences(self, text, max_sentences=5):
        """Chunk large text into smaller segments by grouping sentences."""
        sentences = sent_tokenize(text)
        return [
            " ".join(sentences[i : i + max_sentences])
            for i in range(0, len(sentences), max_sentences)
        ]

    def fit_topic_model(self, text_data):
        """Fit BERTopic model to the text data."""
        if not isinstance(text_data, list):
            raise ValueError("Text data must be a list of text chunks.")

        text_data = [str(text) for text in text_data if text and isinstance(text, str)]

        # Create the topic model with or without OpenAI as the representation model
        self.topic_model = BERTopic(representation_model=self.representation_model)
        topics, probs = self.topic_model.fit_transform(text_data)
        return topics, probs

    def get_topic_info(self):
        """Get the topic information after fitting the model."""
        if self.topic_model:
            return self.topic_model.get_topic_info()
        else:
            return "Topic model has not been fitted yet."

    def assign_topics(self, original_chunks, processed_chunks, topics):
        """
        Map original (unprocessed) text chunks to their topic categories
        based on processed chunk topics.
        """
        categorized_chunks = []

        for i, chunk in enumerate(original_chunks):
            topic_id = topics[i] if i < len(topics) else -1
            if topic_id == -1:
                topic_name, topic_representation, topic_count = "Outlier", [], 0
            else:
                topic_data = self.topic_model.get_topic(topic_id)
                print(topic_data)
                topic_name = f"{topic_id}_{'_'.join(map(str, topic_data[0]))}"
                topic_representation = topic_data

            categorized_chunks.append(
                {
                    "Chunk": chunk,
                    "Processed Chunk": (
                        processed_chunks[i] if i < len(processed_chunks) else ""
                    ),
                    "Topic ID": topic_id,
                    "Topic Name": topic_name,
                    "Topic Representation": topic_representation,
                }
            )

        return categorized_chunks

    def process_text(self, text):
        """Complete pipeline: chunk text, preprocess, fit the topic model, and return topic info."""

        original_chunks = self.chunk_text_by_sentences(text)
        processed_chunks = [
            self.lemmatize_text(self.remove_stopwords(chunk))
            for chunk in original_chunks
            if chunk.strip()
        ]

        print(f"Text split into {len(processed_chunks)} chunks.")

        # Create topics using processed text
        print("Creating topics...")
        topics, probs = self.fit_topic_model(processed_chunks)

        # Map the topics back to the original text
        result_list = self.assign_topics(original_chunks, processed_chunks, topics)

        print("Topic creation complete.")

        return result_list

    def remove_stopwords(self, text):
        """Remove stopwords from text for topic modeling."""
        return " ".join(
            [word for word in text.split() if word.lower() not in ENGLISH_STOP_WORDS]
        )

    def lemmatize_text(self, text):
        """Lemmatize words in the text."""
        lemmatizer = WordNetLemmatizer()
        return " ".join([lemmatizer.lemmatize(word) for word in text.split()])


if __name__ == "__main__":
    # Use Reuters corpus for demonstration
    dataset_text = " ".join(reuters.raw(fileid) for fileid in reuters.fileids()[:50])

    # Initialize processor without OpenAI first
    processor = ThematicTextProcessor()
    labeled_list = processor.process_text(dataset_text)
    with open("data/topics_without_openai.json", "w") as f:
        json.dump(labeled_list, f, indent=4)

    print(f"Categorized text chunks saved to 'topics_without_openai.json'")

    # Initialize processor with OpenAI
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY", "")
    if openai_api_key:
        print("\nUsing OpenAI for topic representation...")
        processor_openai = ThematicTextProcessor(
            api_key=openai_api_key, use_openai=True
        )

        labeled_list = processor_openai.process_text(dataset_text)
        with open("data/topics_with_openai.json", "w") as f:
            json.dump(labeled_list, f, indent=4)
