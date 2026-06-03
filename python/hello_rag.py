import os
import glob
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_anthropic import ChatAnthropic

# 0. Configuración (Equivalente a application.properties)
CHUNK_SIZE = 300
CHUNK_OVERLAP = 30
K = 3
MIN_SCORE = 0.5
CHAT_MODEL_NAME = "claude-haiku-4-5-20251001"  # Claude Haiku
SHOW_PROMPT = True

def require_env(name: str) -> str:
    val = os.getenv(name)
    if not val or not val.strip():
        raise ValueError(f"Variable de entorno '{name}' no definida.")
    return val

def load_documents(corpus_dir: str) -> list[Document]:
    documents = []
    # Busca todos los archivos markdown en el corpus
    for filepath in glob.glob(os.path.join(corpus_dir, "*.md")):
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        # Genera el documento con metadata de nombre de archivo
        doc = Document(
            page_content=content,
            metadata={"file_name": os.path.basename(filepath)}
        )
        documents.append(doc)
    return documents

def main():
    # Asegura la API Key
    api_key = require_env("ANTHROPIC_API_KEY")

    # 1. Inicializar modelos
    # Cargamos el mismo modelo BGE small que usa la versión de Java
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vector_store = InMemoryVectorStore(embeddings)
    chat_model = ChatAnthropic(model=CHAT_MODEL_NAME, api_key=api_key)

    # 2. Fase 1: INDEXACIÓN
    print("Cargando documentos...")
    raw_docs = load_documents("corpus")
    print(f"Cargados {len(raw_docs)} documentos del corpus")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(raw_docs)
    print(f"Generados {len(chunks)} chunks (tamano={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    vector_store.add_documents(chunks)
    print(f"Indexados {len(chunks)} vectores\n")

    # 3. Consultas a realizar
    questions = [
        "Cuanto cuesta enviar a Madrid?",
        "Cuantos dias tengo para devolver un producto?",
        "Aceptan pago contra reembolso a Tenerife?"
    ]

    for question in questions:
        print("=" * 70)
        print(f"PREGUNTA: {question}")
        print("=" * 70)

        # Fase 2: RECUPERACIÓN (búsqueda por similitud con puntuación)
        results = vector_store.similarity_search_with_score(question, k=K)
        # Filtrar por puntuación mínima
        matches = [m for m in results if m[1] >= MIN_SCORE]

        if not matches:
            print(f"Sin fragmentos relevantes (score < {MIN_SCORE}).\n")
            continue

        print("Fragmentos recuperados:")
        for idx, (doc, score) in enumerate(matches):
            preview = doc.page_content.replace("\n", " ")
            if len(preview) > 70:
                preview = preview[:67] + "..."
            print(f"  [{idx + 1}] score={score:.3f}  {preview}")

        # Fase 3: ENSAMBLADO DEL PROMPT
        # Construimos el bloque de contexto
        context_parts = ["<context>\n"]
        for idx, (doc, _) in enumerate(matches):
            source = doc.metadata.get("file_name", "desconocido")
            context_parts.append(f'<doc id="{idx + 1}" source="{source}">\n')
            context_parts.append(doc.page_content + "\n")
            context_parts.append("</doc>\n")
        context_parts.append("</context>\n\n")
        
        prompt = (
            "".join(context_parts)
            + f"<question>\n{question}\n</question>\n\n"
            + "<instructions>\n"
              "Responde a la pregunta planteada basándote ÚNICAMENTE en la información explícita que se encuentra dentro de las etiquetas <context>.\n\n"
              "Si la respuesta no se puede deducir directamente del contexto proporcionado, debes responder estrictamente con la siguiente frase literal: \"No tengo esa información en el corpus.\"\n\n"
              "Es obligatorio que justifiques tu respuesta citando textualmente el id y la fuente del documento que sustenta tu afirmación (por ejemplo: [doc id=\"1\" source=\"archivo.md\"]). Sé conciso y directo en tu respuesta.\n"
              "</instructions>"
        )

        if SHOW_PROMPT:
            print("\n------------------------------ PROMPT ENSAMBLADO ENVIADO AL MODELO ------------------------------")
            print(prompt)
            print("-------------------------------------------------------------------------------------------------")

        # Fase 4: GENERACIÓN
        answer = chat_model.invoke(prompt).content
        print(f"\nRESPUESTA:\n{answer}\n")

if __name__ == "__main__":
    main()
