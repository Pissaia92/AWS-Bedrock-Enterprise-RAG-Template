import os

# Estrutura de pastas do projeto
folders = [
    "app",
    "app/core",       # Lógica principal do RAG
    "app/api",        # Endpoints FastAPI
    "app/ui",         # Interface Streamlit
    "data/raw",       # Onde colocaremos os PDFs
    "vector_store",   # Onde o FAISS salvará o índice local
    "scripts",        # Scripts de ingestão/teste
    "tests"
]

# Arquivos iniciais e seus conteúdos básicos
files = {
    ".gitignore": """
.env
.venv
__pycache__
vector_store/
data/raw/*
!data/raw/.gitkeep
.DS_Store
""",
    ".env": """
# Configurações AWS (Preencha com suas chaves se não tiver AWS CLI configurado)
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_DEFAULT_REGION=us-east-1

# Configurações do Modelo
BEDROCK_LLM_MODEL_ID=anthropic.claude-3-haiku-20240307-v1:0
BEDROCK_EMBEDDING_MODEL_ID=amazon.titan-embed-text-v1
""",
    "data/raw/.gitkeep": "",
    "pyproject.toml": """
[tool.poetry]
name = "aws-genai-rag"
version = "0.1.0"
description = "Enterprise RAG Template using AWS Bedrock and LangChain"
authors = ["Seu Nome <seuemail@example.com>"]

[tool.poetry.dependencies]
python = "^3.10"
boto3 = "^1.34"         # SDK da AWS
langchain = "^0.1"      # Framework de Orquestração
langchain-community = "^0.0.20"
langchain-aws = "^0.1"  # Integração específica AWS
faiss-cpu = "^1.8"      # Vector DB local (Facebook AI Similarity Search)
pypdf = "^4.0"          # Leitor de PDF
python-dotenv = "^1.0"  # Gestão de variáveis de ambiente
fastapi = "^0.109"      # API
uvicorn = "^0.27"       # Servidor
streamlit = "^1.31"     # UI Rápida
tiktoken = "^0.6"       # Tokenizer

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
""",
    "README.md": """
# AWS Bedrock RAG Enterprise Template

Template de arquitetura para soluções de Generative AI utilizando AWS Bedrock.

## Stack
- **LLM**: AWS Bedrock (Claude 3 / Titan)
- **Orchestration**: LangChain
- **Vector Store**: FAISS (Local) / OpenSearch (Prod)
"""
}

def create_structure():
    print("🚀 Iniciando setup do projeto AWS RAG...")
    
    # Criar pastas
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"✅ Pasta criada: {folder}")

    # Criar arquivos
    for filename, content in files.items():
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content.strip())
        print(f"✅ Arquivo criado: {filename}")

    print("\n🏁 Estrutura finalizada! Próximos passos:")
    print("1. Instale o Poetry (se não tiver)")
    print("2. Rode: poetry install")
    print("3. Configure o arquivo .env com suas credenciais AWS")

if __name__ == "__main__":
    create_structure()