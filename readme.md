
---

# Text Analysis Pipeline using LangGraph & Google Gemini LLM



This project implements a powerful **text analysis pipeline** leveraging **LangGraph** for workflow orchestration and **Google Gemini API** as the LLM (Large Language Model) for natural language processing tasks.

## Features

The pipeline consists of three key steps:

1. **Text Classification**: Automatically classifies the input text into one of the following categories: 
   - News
   - Blog
   - Research
   - Other

2. **Entity Extraction**: Extracts key entities from the text, including:
   - Person
   - Organization
   - Location

3. **Text Summarization**: Provides a concise one-sentence summary of the input text.

## How It Works

- **LangGraph's StateGraph** is used to build the pipeline as a series of nodes, each representing a specific task (classification, entity extraction, summarization).
- **Google Gemini** API is utilized as the LLM to handle the natural language processing for each task, ensuring accurate and fast text analysis.

## Pipeline Overview

The pipeline is defined as follows:

- Start → **Text Classification** → **Entity Extraction** → **Summarization** → End

The pipeline flow is managed using LangGraph's stateful graph execution, ensuring a smooth transition between each task.

## Dependencies

- **LangGraph**
- **LangChain**
- **Google Gemini API**
- **Python 3.x**
- Additional Python libraries listed in `pyproject.toml`.

## Usage

1. Set up your environment and install dependencies.
2. Set your Google Gemini API key.
3. Run the pipeline to classify, extract entities, and summarize any input text.

```bash
poetry add (all packages with spaces mention here)
poetry run python main.py
```

## Conclusion

This project showcases a scalable approach to performing common NLP tasks like text classification, entity extraction, and summarization using cutting-edge LLMs and a structured, maintainable graph-based execution model.

---

