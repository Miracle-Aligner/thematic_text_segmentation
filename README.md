
# Thematic Text Processing with BERTopic

This project provides a tool for performing thematic text processing using the BERTopic model and OpenAI's GPT-based representations. It allows users to process text data, create topics, and visualize or download the results.
You can find live demo [HERE](https://textsegmentation.streamlit.app/).

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Requirements](#requirements)

## Overview

This project demonstrates how to use the BERTopic model to process text data and categorize it into various topics. It integrates two key approaches:
- Using **BERTopic** for topic modeling.
- Optionally leveraging **OpenAI**'s API for enhanced topic representation.

It supports both the Reuters dataset and custom text input through a Streamlit web interface. Users can view the generated topics and download them as a JSON file.

## Installation

To get started with the project, follow these steps:

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. Install the required Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Ensure you have an OpenAI API key (if using OpenAI for topic representation).

   - Set your OpenAI API key in a `.env` file:
   
     ```bash
     OPENAI_API_KEY=<your-api-key>
     ```

## Usage

### 1. Running the Streamlit App

To run the Streamlit app, execute the following command:

```bash
streamlit run streamlit_app.py
```

- The app allows you to either use the Reuters dataset or provide your own text to create topics.
- You can choose whether to use OpenAI for enhanced topic representation via a checkbox in the sidebar.
- After processing, topics will be displayed, and you can download the result as a JSON file.

### 2. Running the Text Processor Directly

To process the text data programmatically (e.g., without the Streamlit app), you can use the `text_processor.py` file. This file demonstrates how to use the `ThematicTextProcessor` class to fit a BERTopic model and create topics.

For example:

```python
from text_processor import ThematicTextProcessor

processor = ThematicTextProcessor()
dataset_text = " ".join(reuters.raw(fileid) for fileid in reuters.fileids()[:50])
result = processor.process_text(dataset_text)

# Save the result to a JSON file
with open("data/topics_without_openai.json", "w") as f:
    json.dump(result, f, indent=4)
```

## File Structure

```
.
├── README.md
├── __pycache__
│   └── text_processor.cpython-311.pyc
├── data
│   ├── topics_with_openai.json
│   └── topics_without_openai.json
├── requirements.txt
├── streamlit_app.py
└── text_processor.py
```

- `README.md`: Documentation for the project.
- `__pycache__/`: Cached Python files.
- `data/`: Contains the output files with generated topics.
- `requirements.txt`: A list of dependencies to install.
- `streamlit_app.py`: The Streamlit web application for interactive topic modeling.
- `text_processor.py`: The Python script that processes text, creates topics, and saves the results.

## Requirements

To run this project, the following Python packages are required:

- `python-dotenv`: For loading environment variables (e.g., OpenAI API key).
- `openai`: For accessing OpenAI's API (optional).
- `bertopic`: For topic modeling.
- `scikit-learn`: For text preprocessing and stop word removal.
- `nltk`: For tokenization and lemmatization.
- `streamlit`: For the web interface.

Install all dependencies using:

```bash
pip install -r requirements.txt
```
