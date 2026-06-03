# mod5-rag-hello-world

Pipeline RAG mínimo en Java con LangChain4j. Implementa las cuatro fases del módulo 5 del curso *IA generativa en el desarrollo de software* en una sola clase, con un corpus de ejemplo incluido.

---

## Versiones del proyecto

Este repositorio contiene dos implementaciones del mismo pipeline RAG para facilitar el aprendizaje progresivo:

| Rama | Enfoque | Ideal para... |
|---|---|---|
| `main` | **Java Puro (POJO)** | Entender las bases sin abstracciones. Todo se inicializa manualmente. |
| `feature/spring-boot-migration` | **Spring Boot 3** | Ver cómo escalar el pipeline a una aplicación profesional y empresarial. |

Para cambiar a la versión de Spring Boot:
```bash
git checkout feature/spring-boot-migration
```

---

## Requisitos

| Herramienta | Versión mínima |
|---|---|
| JDK | 21 |
| Maven | 3.9 |

**API key de Anthropic.** Necesitas una clave con acceso a Claude Haiku.

---

## Configuración

```bash
# 1. Clonar
git clone https://github.com/leroidubuffet/mod5-rag-hello-world.git
cd mod5-rag-hello-world

# 2. Exportar la API key (nunca la pongas en el código ni en el pom.xml)
export ANTHROPIC_API_KEY=sk-ant-...

# 3. Compilar y ejecutar
mvn -q exec:java
```

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
Generados 5 chunks (tamano=300, overlap=30)
Indexados 5 vectores

======================================================================
PREGUNTA: Cuanto cuesta enviar a Madrid?
======================================================================
Fragmentos recuperados:
  [1] score=0.612  src=envios.md          # Politica de envios de Casa Tortuga ...
  [2] score=0.523  src=envios.md          ### Baleares y Canarias - Baleares: 8...
  [3] score=0.501  src=envios.md          Los pedidos con importe superior a 49...

RESPUESTA:
El envío a Madrid cuesta 4,90 EUR para paquetes de hasta 5 kg [doc 1].

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
| Chunks grandes | `CHUNK_SIZE = 1000` | Los scores bajan; la señal se diluye entre tokens irrelevantes |
| Chunks pequeños | `CHUNK_SIZE = 80` | Los chunks pierden contexto; el modelo falla en preguntas que cruzan frases |
| Top-1 | `K = 1` | Si el top-1 es incorrecto, el sistema falla sin alternativa |
| Filtro estricto | `MIN_SCORE = 0.8` | El sistema responde "no tengo esa información" para la mayoría de preguntas |
| Pregunta fuera de corpus | Añadir `"Cuantos rinocerontes hay en Africa?"` a la lista | El modelo debe responder con la frase literal del prompt |

---

## Variante con ChromaDB

Por defecto el store es en memoria (`InMemoryEmbeddingStore`): los vectores no persisten entre ejecuciones. Para usar ChromaDB con persistencia real:

**1. Levantar ChromaDB:**

```bash
docker compose up -d
curl http://localhost:8000/api/v1/heartbeat   # debe responder
```

**2. Descomentar la dependencia en `pom.xml`:**

```xml
<dependency>
    <groupId>dev.langchain4j</groupId>
    <artifactId>langchain4j-chroma</artifactId>
    <version>${langchain4j.version}</version>
</dependency>
```

**3. Sustituir el store en `HelloRag.java`:**

```java
// Antes:
EmbeddingStore<TextSegment> store = new InMemoryEmbeddingStore<>();

// Después:
EmbeddingStore<TextSegment> store = ChromaEmbeddingStore.builder()
        .baseUrl("http://localhost:8000")
        .collectionName("casa-tortuga")
        .build();
```

**4. Añadir el import:**

```java
import dev.langchain4j.store.embedding.chroma.ChromaEmbeddingStore;
```

**5. Reejecutar:**

```bash
mvn -q exec:java
```

El comportamiento es idéntico desde el código. La diferencia es que ahora los vectores persisten en el volumen Docker entre ejecuciones y se pueden inspeccionar vía la API HTTP de ChromaDB (`GET http://localhost:8000/api/v1/collections`).

Esto ilustra la promesa de la abstracción `EmbeddingStore` de LangChain4j: cambiar la implementación del store no requiere tocar el resto del pipeline.

---

## Qué NO hace este proyecto

- **Sin re-ranking.** La recuperación va directa a top-k por similitud coseno.
- **Sin query rewriting** ni decisión de cuándo no recuperar (recupera siempre).
- **Sin evaluación automática.** La calidad se juzga leyendo las respuestas. La evaluación automatizada con métricas de RAG se verá en el módulo 8.

Estas extensiones son el paso natural siguiente. La [documentación de LangChain4j sobre RAG](https://docs.langchain4j.dev/tutorials/rag) y el [Anthropic cookbook](https://github.com/anthropics/anthropic-cookbook) las cubren en detalle.

---

## Estructura del proyecto

```
mod5-rag-hello-world/
├── corpus/
│   ├── envios.md          políticas de envío de Casa Tortuga
│   ├── devoluciones.md    políticas de devolución
│   └── pagos.md           métodos de pago
├── src/main/java/com/example/rag/
│   └── HelloRag.java      pipeline completo en una clase (~200 líneas)
├── docker-compose.yml     variante ChromaDB (opcional)
└── pom.xml
```

## Licencia

MIT
