# mod5-rag-hello-world

Pipeline RAG mínimo implementado en **Java (con LangChain4j)** y **Python (con LangChain)**. Implementa las cuatro fases del módulo 5 del curso *IA generativa en el desarrollo de software*, utilizando un corpus de ejemplo compartido.

## Organización del Repositorio (Ramas)

Este proyecto está organizado en ramas para separar y facilitar el aprendizaje de cada tecnología de manera aislada:

*   **[`main`](https://github.com/leroidubuffet/mod5-rag-hello-world/tree/main)** (esta rama): Implementación en **Spring Boot 3 + LangChain4j**.
*   **[`python`](https://github.com/leroidubuffet/mod5-rag-hello-world/tree/python)**: Implementación en **Python 3 + LangChain**.

Cada rama contiene de forma exclusiva los archivos y la documentación correspondiente a su tecnología.

---

# Spring Boot 3 + LangChain4j — Rama `main`

Esta rama contiene la versión con Spring Boot 3. Los modelos, el store de embeddings y los servicios se gestionan como beans Spring, lo que facilita la configuración y la extensión del pipeline.

## Funcionalidades del Programa

Esta implementación cuenta con **doble funcionalidad**:

1. **Modo Lote (Batch / Preguntas prefijadas)**:
   - Carga e indexa los documentos del corpus en memoria.
   - Realiza de forma automática tres preguntas predefinidas sobre las políticas de la tienda y muestra los fragmentos relevantes junto con las respuestas de Claude Haiku citando sus fuentes.

2. **Modo Chat Interactivo (CLI)**:
   - Abre un canal de comunicación en tiempo real en la terminal para realizar preguntas libres sobre el corpus.
   - **Memoria de turnos**: Mantiene el contexto de las respuestas y preguntas previas.
   - **Condensación de Consultas**: Toma tu nueva pregunta y el historial previo para generar una consulta optimizada para la base de datos vectorial mediante una llamada previa a Claude Haiku.

## Requisitos

- **API key de Anthropic**: Necesitas una clave con acceso a Claude Haiku.
- **JDK 21** y **Maven 3.9+**

## Configuración y Ejecución

```bash
# 1. Clonar
git clone https://github.com/leroidubuffet/mod5-rag-hello-world.git
cd mod5-rag-hello-world

# 2. Exportar la API key (nunca la pongas en el código ni en el pom.xml)
export ANTHROPIC_API_KEY=sk-ant-...
```

Además, puedes modificar las instrucciones del sistema editando el archivo [system_prompt.txt](./system_prompt.txt) en la raíz del repositorio.

### 1. Ejecutar Modo Lote (Batch Questions)
```bash
mvn spring-boot:run -q
```

### 2. Ejecutar Modo Chat Interactivo
```bash
mvn spring-boot:run -q -Dspring-boot.run.arguments="--rag.mode=chat"
```
Escribe `salir`, `exit` o `quit` en la terminal para cerrar la sesión.

La primera ejecución descarga el modelo de embeddings BGE (~25 MB como recurso del JAR vía Maven). Las siguientes son inmediatas.

---

## Qué hace

El programa recorre tres preguntas sobre las políticas ficticias de "Casa Tortuga" y para cada una ejecuta las cuatro fases del pipeline:

```
Fase 1 — INDEXACIÓN (una vez al arrancar)
  corpus/*.md → chunks → embeddings → store en memoria

Fase 2 — RECUPERACIÓN (por pregunta)
  pregunta → embedding → búsqueda top-k en el store

Fase 3 — ENSAMBLADO
  fragmentos recuperados + pregunta → prompt con bloques XML

Fase 4 — GENERACIÓN
  prompt → Claude Haiku → respuesta con cita de fuente
```

Las tres preguntas están diseñadas para cubrir casos distintos:

| Pregunta | Caso |
|---|---|
| ¿Cuánto cuesta enviar a Madrid? | Respuesta directa en un documento |
| ¿Cuántos días tengo para devolver un producto? | Respuesta en otro documento |
| ¿Aceptan pago contra reembolso a Tenerife? | Capciosa: el reembolso existe pero no aplica a Canarias |

### Output esperado

```
Cargados 3 documentos del corpus
Generados 20 chunks (tamano=300, overlap=30)
Indexados 20 vectores

======================================================================
PREGUNTA: Cuanto cuesta enviar a Madrid?
======================================================================
Fragmentos recuperados:
  [1] score=0.852  ## Envio gratuito  Los pedidos con importe superior a 49 EUR (sin c...
  [2] score=0.847  ## Como solicitar la devolucion
  [3] score=0.835  - Madrid, Barcelona, Valencia, Sevilla: 4,90 EUR para paquetes de h...

RESPUESTA:
Según la información proporcionada, enviar a Madrid cuesta **4,90 EUR para paquetes de hasta 5 kg**. [doc id="3" source="envios.md"]

...
```

Los scores varían entre ejecuciones porque dependen del chunking y del modelo de embedding; el texto exacto de las respuestas varía porque el LLM no es determinista.

---

## Experimentos sugeridos

El archivo `src/main/resources/application.properties` es el punto de entrada para experimentar sin tocar el código:

```properties
rag.chunk.size=300
rag.chunk.overlap=30
rag.retrieval.k=3
rag.retrieval.min_score=0.5
```

| Experimento | Cambio | Qué observar |
|---|---|---|
| Chunks grandes | `rag.chunk.size=1000` | Los scores bajan; la señal se diluye entre tokens irrelevantes |
| Chunks pequeños | `rag.chunk.size=80` | Los chunks pierden contexto; el modelo falla en preguntas que cruzan frases |
| Top-1 | `rag.retrieval.k=1` | Si el top-1 es incorrecto, el sistema falla sin alternativa |
| Filtro estricto | `rag.retrieval.min_score=0.8` | El sistema responde "no tengo esa información" para la mayoría de preguntas |
| Pregunta fuera de corpus | Añadir `"Cuantos rinocerontes hay en Africa?"` a la lista | El modelo debe responder con la frase literal del prompt |

---

## Variante con ChromaDB

Por defecto el store es en memoria (`InMemoryEmbeddingStore`): los vectores no persisten entre ejecuciones. Para usar ChromaDB con persistencia real:

**1. Levantar ChromaDB:**

```bash
docker compose up -d
curl http://localhost:8000/api/v1/heartbeat   # debe responder
```

**2. Añadir la dependencia en `pom.xml`:**

```xml
<dependency>
    <groupId>dev.langchain4j</groupId>
    <artifactId>langchain4j-chroma</artifactId>
    <version>${langchain4j.version}</version>
</dependency>
```

**3. Sustituir el store en `RagConfig.java`:**

```java
// Antes:
@Bean
public EmbeddingStore<TextSegment> embeddingStore() {
    return new InMemoryEmbeddingStore<>();
}

// Después:
@Bean
public EmbeddingStore<TextSegment> embeddingStore() {
    return ChromaEmbeddingStore.builder()
            .baseUrl("http://localhost:8000")
            .collectionName("casa-tortuga")
            .build();
}
```

**4. Añadir el import en `RagConfig.java`:**

```java
import dev.langchain4j.store.embedding.chroma.ChromaEmbeddingStore;
```

**5. Reejecutar:**

```bash
mvn spring-boot:run -q
```

El comportamiento es idéntico desde el código. La diferencia es que ahora los vectores persisten en el volumen Docker entre ejecuciones y se pueden inspeccionar vía la API HTTP de ChromaDB (`GET http://localhost:8000/api/v1/collections`).

## Flujo del RAG (4 Fases)
1. **Fase 1: Indexación (al arrancar)**: Carga los documentos markdown de la carpeta `/corpus`, los divide en chunks de 300 caracteres (overlap de 30), genera embeddings locales con el modelo BGE-small-en-v1.5-q y los guarda en un store en memoria.
2. **Fase 2: Recuperación (por pregunta)**: Genera el embedding de la pregunta y busca los top-3 fragmentos más similares.
3. **Fase 3: Ensamblado**: Agrupa los fragmentos recuperados con la pregunta en un prompt delimitado por etiquetas XML.
4. **Fase 4: Generación**: Envía el prompt a Claude Haiku y retorna la respuesta final justificando la fuente.
