# mod5-rag-hello-world

Pipeline RAG mínimo implementado en **Java (con LangChain4j)** y **Python (con LangChain)**. Implementa las cuatro fases del módulo 5 del curso *IA generativa en el desarrollo de software*, utilizando un corpus de ejemplo compartido.

---

## Estructura del repositorio

Este repositorio está organizado como un monorrep para permitir la comparación directa y el aprendizaje progresivo de las diferentes tecnologías:

| Carpeta | Enfoque | Tecnología | Características |
|---|---|---|---|
| [`java-pojo/`](./java-pojo) | **Java Puro (POJO)** | Java 21 + Maven | Todo se inicializa manualmente. Ideal para entender el flujo básico sin abstracciones. |
| [`java-springboot/`](./java-springboot) | **Spring Boot 3** | Java 21 + Spring Boot 3 | Estructura empresarial con inyección de dependencias, beans de configuración y properties externalizados. |
| [`python/`](./python) | **Python 3** | Python 3.10+ + LangChain | Implementación ágil y directa con las bibliotecas estándar de LangChain en Python. |

---

## Requisitos

- **API key de Anthropic**: Necesitas una clave con acceso a Claude Haiku.
- **JDK 21** y **Maven 3.9+** (para las versiones de Java).
- **Python 3.10+** (para la versión de Python).

---

## Configuración y Ejecución

Primero, exporta tu clave de API en la terminal:
```bash
export ANTHROPIC_API_KEY=sk-ant-your-key-here
```

### Variante 1: Java Puro (POJO)
```bash
cd java-pojo
mvn clean compile exec:java
```

### Variante 2: Spring Boot 3
```bash
cd java-springboot
mvn clean compile spring-boot:run
```

### Variante 3: Python
```bash
cd python
# Crear y activar entorno virtual
python3 -m venv venv
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar el script
python hello_rag.py
```

---

## Qué hace el programa

El pipeline procesa tres preguntas sobre las políticas internas de la tienda de ejemplo **"Casa Tortuga"**:
1. *¿Cuánto cuesta enviar a Madrid?* (Respuesta directa en un documento de envíos)
2. *¿Cuántos días tengo para devolver un producto?* (Respuesta directa en políticas de devoluciones)
3. *¿Aceptan pago contra reembolso a Tenerife?* (Pregunta capciosa: el reembolso existe pero no aplica a Canarias)

### Flujo del RAG (4 Fases)
1. **Fase 1 — Indexación (al arrancar)**: Carga los documentos markdown de la carpeta raíz `/corpus`, los divide en chunks de 300 caracteres (overlap de 30), genera embeddings locales con el modelo BGE-small-en-v1.5 y los guarda en un store en memoria.
2. **Fase 2 — Recuperación (por pregunta)**: Genera el embedding de la pregunta y busca los top-3 fragmentos más similares.
3. **Fase 3 — Ensamblado**: Agrupa los fragmentos recuperados con la pregunta en un prompt delimitado por etiquetas XML.
4. **Fase 4 — Generación**: Envía el prompt a Claude Haiku y retorna la respuesta final justificando la fuente.

---

## Variante con ChromaDB (Persistencia Real)

Por defecto, todas las implementaciones usan almacenamiento en memoria (`InMemoryEmbeddingStore`). Para persistir los vectores usando **ChromaDB**:

### 1. Levantar ChromaDB localmente
```bash
docker compose up -d
curl http://localhost:8000/api/v1/heartbeat   # debe responder
```

### 2. Configurar el código según tu variante:

#### En Java POJO (`java-pojo`):
- Descomenta la dependencia `langchain4j-chroma` en `java-pojo/pom.xml`.
- Sustituye la instanciación de `store` en `java-pojo/src/main/java/com/example/rag/HelloRag.java` para usar `ChromaEmbeddingStore.builder()`.

#### En Java Spring Boot (`java-springboot`):
- Descomenta o añade la dependencia `langchain4j-chroma` en `java-springboot/pom.xml`.
- Modifica el `@Bean` de `embeddingStore()` en `java-springboot/src/main/java/com/example/rag/RagConfig.java`:
  ```java
  @Bean
  public EmbeddingStore<TextSegment> embeddingStore() {
      return ChromaEmbeddingStore.builder()
              .baseUrl("http://localhost:8000")
              .collectionName("casa-tortuga")
              .build();
  }
  ```

#### En Python (`python`):
- Instala la biblioteca de ChromaDB (`pip install chromadb`).
- Sustituye el uso de `InMemoryVectorStore` en `python/hello_rag.py` por el cliente de ChromaDB de LangChain:
  ```python
  from langchain_chroma import Chroma
  vector_store = Chroma(
      collection_name="casa-tortuga",
      embedding_function=embeddings,
      persist_directory="./chroma_db"
  )
  ```

---

## Licencia

MIT
