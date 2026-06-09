import os
import glob
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb
from langchain_chroma import Chroma
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# 0. Configuración básica
CHUNK_SIZE = 300
CHUNK_OVERLAP = 30
K = 3
MIN_SCORE = 0.5
CHAT_MODEL_NAME = "claude-haiku-4-5-20251001"

def require_env(name: str) -> str:
    val = os.getenv(name)
    if not val or not val.strip():
        raise ValueError(f"Variable de entorno '{name}' no definida.")
    return val

def load_documents(corpus_dir: str) -> list[Document]:
    documents = []
    for filepath in glob.glob(os.path.join(corpus_dir, "*.md")):
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        doc = Document(
            page_content=content,
            metadata={"file_name": os.path.basename(filepath)}
        )
        documents.append(doc)
    return documents

def condense_query(chat_history, latest_question, chat_model) -> str:
    """
    Toma el historial del chat y la última pregunta del usuario, y genera
    una consulta de búsqueda independiente que contiene todo el contexto.
    """
    if not chat_history:
        return latest_question

    history_parts = []
    for msg in chat_history:
        role = "Usuario" if isinstance(msg, HumanMessage) else "Asistente"
        # Limpiamos posibles metadatos del prompt del usuario
        content = msg.content
        if "Pregunta del usuario:" in content:
            content = content.split("Pregunta del usuario:")[-1].strip()
        history_parts.append(f"{role}: {content}")
    history_str = "\n".join(history_parts)

    condense_prompt = (
        "Dado el siguiente historial de conversación y una nueva pregunta del usuario, "
        "crea una consulta de búsqueda independiente y optimizada en español para buscar en una base de datos vectorial.\n"
        "La consulta debe ser concisa y resolver pronombres o referencias contextuales.\n\n"
        f"Historial de conversación:\n{history_str}\n\n"
        f"Nueva pregunta: {latest_question}\n\n"
        "Consulta de búsqueda optimizada (responde SOLO con la consulta):"
    )
    
    condensed_query = chat_model.invoke(condense_prompt).content.strip()
    # Limpiamos comillas si el modelo las añade
    condensed_query = condensed_query.strip('"').strip("'")
    return condensed_query

def build_context_str(matches) -> str:
    context_parts = []
    for idx, (doc, _) in enumerate(matches):
        source = doc.metadata.get("file_name", "desconocido")
        context_parts.append(f'<doc id="{idx + 1}" source="{source}">\n')
        context_parts.append(doc.page_content + "\n")
        context_parts.append("</doc>\n")
    return "".join(context_parts)

def main():
    api_key = require_env("ANTHROPIC_API_KEY")

    # 1. Inicializar modelos
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    chroma_client = chromadb.HttpClient(host="localhost", port=8000)
    chroma_client.delete_collection("casa-tortuga")
    vector_store = Chroma(
        collection_name="casa-tortuga",
        embedding_function=embeddings,
        client=chroma_client,
        collection_metadata={"hnsw:space": "cosine"},
    )
    chat_model = ChatAnthropic(model=CHAT_MODEL_NAME, api_key=api_key, temperature=0)

    # 2. Ingesta e indexación de documentos
    script_dir = os.path.dirname(os.path.abspath(__file__))
    corpus_dir = os.path.join(script_dir, "..", "corpus")
    
    print("Cargando y procesando corpus...")
    raw_docs = load_documents(corpus_dir)
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(raw_docs)
    vector_store.add_documents(chunks)
    print(f"Indexados {len(chunks)} vectores en ChromaDB.")
    print("¡Sistema RAG listo! Escribe 'salir' o 'exit' para terminar.\n")

    # 3. Inicializar historial de chat y leer prompt de sistema
    chat_history = []
    system_prompt_path = os.path.join(script_dir, "..", "system_prompt.txt")
    try:
        with open(system_prompt_path, "r", encoding="utf-8") as f:
            system_instruction = f.read().strip()
    except Exception as e:
        raise FileNotFoundError(f"No se pudo leer el archivo de prompt de sistema en {system_prompt_path}: {e}")

    # Le añadimos la definición de persona al inicio
    system_instruction = f"Eres un asistente virtual de atención al cliente de 'Casa Tortuga'.\n\n{system_instruction}"

    # Bucle de conversación
    while True:
        try:
            user_input = input("Tú: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["salir", "exit", "quit"]:
                print("Asistente: ¡Hasta luego!")
                break

            # A. Condensar pregunta si hay historial
            search_query = condense_query(chat_history, user_input, chat_model)
            if search_query != user_input:
                print(f"\n[Búsqueda RAG optimizada: '{search_query}']")

            # B. Recuperación de fragmentos
            results = vector_store.similarity_search_with_relevance_scores(search_query, k=K)
            matches = [m for m in results if m[1] >= MIN_SCORE]

            # C. Construir contexto
            context_str = build_context_str(matches)

            # D. Construir lista de mensajes para el LLM
            messages = [SystemMessage(content=system_instruction)]
            # Añadimos el historial previo
            messages.extend(chat_history)
            # Añadimos el mensaje actual estructurado con el contexto RAG de este turno
            current_prompt = (
                f"<context>\n{context_str}</context>\n\n"
                f"Pregunta del usuario: {user_input}"
            )
            messages.append(HumanMessage(content=current_prompt))

            # E. Generación de respuesta
            print("Asistente: Pensando...", end="\r")
            response = chat_model.invoke(messages)
            answer = response.content
            print(f"Asistente: {answer}\n")

            # F. Actualizar historial de chat (usando la pregunta original limpia y la respuesta)
            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=answer))

        except KeyboardInterrupt:
            print("\nAsistente: ¡Hasta luego!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")

if __name__ == "__main__":
    main()
