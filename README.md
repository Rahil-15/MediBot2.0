
## MediBot
# 🩺 About MediBot

MediBot is an AI-powered medical information chatbot built using a Retrieval-Augmented Generation (RAG) pipeline.
It provides educational, non-diagnostic, and source-supported medical information using embeddings, vector search, and LLMs.

⚠️ Disclaimer: MediBot does not provide medical advice or prescriptions.
It is strictly for learning and informational purposes.

🔑 API Usage Transparency

This project uses OpenRouter’s free API access only.
✔ No paid APIs were purchased
✔ The entire project runs on free-tier developer keys
✔ Fully cost-efficient and student-friendly

# How to run?
### STEPS:

clone the repository

```bash
project repo: https://github.com/Rahil-15/MediBot2.0.git
```
### STEP 01: Create a conda environment after opening the repository

```bash
conda create -n MediBot2.0 python=3.10 -y
```

```bash
conda activate MediBot2.0
```

### STEP 02: Install the requirements
```bash
pip install -r requirements.txt
```

## Creata a `.env` file in the root directory and add your pinecone & openAI credentials as follows

```ini
PINECONE_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxx"
OPENROUTER_API_KEY ="xxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

```bash
#run the following command to store the embeddings to pinecone
python store_index.py
```

```bash
#finally run the following command
python app.py
```

### Techstach used:

- Python
- Langchain
- Flask
- GPT
- Pinecone
