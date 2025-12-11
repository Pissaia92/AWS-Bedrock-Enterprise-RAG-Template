import sys
import os
import json
from typing import Any, List, Optional, Dict

import boto3
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# --- IMPORTS ---
try:
    from langchain_core.language_models.llms import LLM
    from langchain_core.callbacks.manager import CallbackManagerForLLMRun
    from langchain_community.vectorstores import FAISS
    from langchain_aws import BedrockEmbeddings
    from langchain_core.prompts import PromptTemplate
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
except ImportError as e:
    print(f"\n❌ ERRO DE IMPORT: {e}")
    sys.exit(1)

# --- CLASSE CUSTOMIZADA (A SOLUÇÃO DEFINITIVA) ---
class TitanLLM(LLM):
    """
    Um wrapper customizado e direto para o Amazon Titan via Boto3.
    Resolve problemas de formatação automática de bibliotecas externas.
    """
    client: Any = None
    model_id: str = "amazon.titan-text-express-v1"
    temperature: float = 0.1
    max_tokens: int = 512

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        # Construção MANUAL do Payload (Garantia de que está certo)
        # O Titan exige "inputText" e "textGenerationConfig"
        payload = {
            "inputText": prompt,
            "textGenerationConfig": {
                "temperature": self.temperature,
                "maxTokenCount": self.max_tokens,
                "stopSequences": stop or [],
                "topP": 0.9
            }
        }

        try:
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(payload),
                accept="application/json",
                contentType="application/json",
            )
            response_body = json.loads(response.get("body").read())
            # O Titan retorna a resposta dentro de 'results'[0]['outputText']
            return response_body.get("results")[0].get("outputText")
        except Exception as e:
            raise ValueError(f"Erro ao chamar Bedrock Titan: {e}")

    @property
    def _llm_type(self) -> str:
        return "amazon_titan_custom"

# --- API SETUP ---
app = FastAPI(title="AWS Bedrock RAG API (Custom)", version="FINAL")

BEDROCK_REGION = "us-east-1"
EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v1"

rag_chain = None

try:
    print("🔄 Inicializando RAG com Wrapper Customizado...")
    
    # 1. Cliente Boto3 (Compartilhado)
    bedrock_client = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)
    
    # 2. Embeddings (Isso funciona bem via biblioteca padrão)
    embeddings = BedrockEmbeddings(
        client=bedrock_client, 
        model_id=EMBEDDING_MODEL_ID
    )

    # 3. Vector Store
    if os.path.exists("vector_store"):
        print("📂 Carregando Vector Store...")
        vectorstore = FAISS.load_local(
            "vector_store", 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        
        # 4. LLM -> USANDO NOSSA CLASSE MANUAL
        print("🤖 Configurando LLM (Titan Custom)...")
        llm = TitanLLM(
            client=bedrock_client,
            temperature=0.1,
            max_tokens=512
        )

        # 5. Prompt & Chain
        template = """
        Você é um assistente útil. Use o contexto abaixo para responder à pergunta.
        Se não souber, diga "Não sei com base no contexto".

        Contexto:
        {context}

        Pergunta: {input}
        
        Resposta:"""
        
        prompt = PromptTemplate(template=template, input_variables=["context", "input"])
        document_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, document_chain)
        
        print("✅ API PRONTA! (Modo Prova de Falhas)")
        
    else:
        print("⚠️  AVISO: Pasta 'vector_store' não encontrada. Rode o ingest primeiro.")

except Exception as e:
    print(f"❌ Erro fatal: {e}")

# --- ENDPOINTS ---
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]

@app.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    if not rag_chain:
        raise HTTPException(status_code=500, detail="RAG não inicializado.")
    
    try:
        response = rag_chain.invoke({"input": request.query})
        
        return QueryResponse(
            answer=response.get("answer", "").strip(),
            sources=list(set([doc.metadata.get("source", "unknown") for doc in response.get("context", [])]))
        )
    except Exception as e:
        print(f"Erro: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)