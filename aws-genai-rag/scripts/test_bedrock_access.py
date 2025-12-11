import boto3
import json

def test_bedrock_access():
    print("🕵️ Testando acesso ao AWS Bedrock...")
    
    # Cria o cliente do Bedrock Runtime (usado para inferência)
    # Certifique-se que suas credenciais AWS estão configuradas no ambiente
    try:
        client = boto3.client("bedrock-runtime", region_name="us-east-1")
    except Exception as e:
        print(f"❌ Erro ao configurar cliente AWS: {e}")
        return

    # TESTE 1: Amazon Titan Embeddings (Essencial para o RAG)
    print("\n1️⃣  Testando Modelo de Embeddings (Amazon Titan)...")
    try:
        response = client.invoke_model(
            modelId="amazon.titan-embed-text-v1",
            body=json.dumps({"inputText": "Teste de acesso ao Bedrock"}),
            contentType="application/json",
            accept="application/json"
        )
        print("✅ SUCESSO! Embeddings do Titan estão funcionando.")
    except Exception as e:
        print(f"❌ Falha no Titan Embeddings: {e}")

    # TESTE 2: Anthropic Claude 3 (O melhor modelo, mas pode ter bloqueio)
    print("\n2️⃣  Testando LLM (Anthropic Claude 3 Haiku)...")
    try:
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Olá, você está funcionando?"}]
        })
        response = client.invoke_model(
            modelId="anthropic.claude-3-haiku-20240307-v1:0",
            body=body,
            contentType="application/json",
            accept="application/json"
        )
        result = json.loads(response['body'].read())
        print(f"✅ SUCESSO! Claude respondeu: {result['content'][0]['text']}")
    except Exception as e:
        print(f"⚠️ Aviso: Claude falhou (provavelmente precisa de use-case). Erro: {e}")
        print("   -> Não tem problema! Se falhar, usaremos o 'amazon.titan-text-express-v1'.")

    # TESTE 3 (Fallback): Amazon Titan Text (Caso o Claude falhe)
    if "Anthropic" not in str(locals().get("result", "")): 
        print("\n3️⃣  Testando Fallback (Amazon Titan Text)...")
        try:
            body = json.dumps({
                "inputText": "Olá, responda com 'Estou vivo'.",
                "textGenerationConfig": {"maxTokenCount": 50, "temperature": 0}
            })
            response = client.invoke_model(
                modelId="amazon.titan-text-express-v1",
                body=body,
                contentType="application/json",
                accept="application/json"
            )
            print("✅ SUCESSO! Amazon Titan Text está funcionando.")
        except Exception as e:
            print(f"❌ Falha no Amazon Titan Text: {e}")

if __name__ == "__main__":
    test_bedrock_access()