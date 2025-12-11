import os
import sys
import boto3

# --- IMPORTS MODERNOS (Compatíveis com requirements.txt atual) ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_aws import BedrockEmbeddings
from langchain_community.vectorstores import FAISS

def ingest_data():
    print("🚀 Iniciando Ingestão de Dados (Stack Moderna)...")
    
    # 1. Configurar Bedrock
    bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")
    embeddings = BedrockEmbeddings(
        client=bedrock_client,
        model_id="amazon.titan-embed-text-v1" 
    )

    # 2. Carregar o PDF
    pdf_path = "data/raw/doc_manual.pdf"
    if not os.path.exists(pdf_path):
        print(f"❌ Erro: Arquivo {pdf_path} não encontrado.")
        return

    print(f"📂 Carregando arquivo: {pdf_path}...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    # 3. Dividir o Texto
    print("✂️  Dividindo documento...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = text_splitter.split_documents(documents)
    print(f"   -> {len(docs)} chunks gerados.")

    # 4. Gerar Embeddings e Salvar
    print("🧠 Gerando Embeddings (FAISS)...")
    try:
        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local("vector_store")
        print("✅ Sucesso! Novo índice 'vector_store' criado com compatibilidade V2.")
        
    except Exception as e:
        print(f"❌ Erro na geração: {e}")

if __name__ == "__main__":
    ingest_data()