# MechAI ⚙️

**Live Demo:** [mechanical-ta-rag-based-llm.streamlit.app](https://mechanical-ta-rag-based-llm.streamlit.app)

An intelligent mechanical engineering assistant that answers questions with accurate, context-aware responses and LaTeX-formatted math equations.

## Features

- Ask any mechanical engineering question and get precise answers
- Mathematical expressions rendered in proper LaTeX format
- Dark ChatGPT-style interface
- Fast responses powered by state-of-the-art AI

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/MechAI.git
cd MechAI
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Copy `.env.example` to `.env` and fill in your API keys:

```
GROQ_API_KEY=your_groq_api_key
PINECONE_API_KEY=your_pinecone_api_key
DB_INDEX_NAME=your_index_name
```

### 4. Run the app

```bash
streamlit run home.py
```

The app will be available at `http://localhost:8501`

## License

This project is licensed under the [MIT License](LICENSE).
