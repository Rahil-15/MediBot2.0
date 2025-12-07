# MediBot-


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
