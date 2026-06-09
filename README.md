# mod5-rag-hello-world

Pipeline RAG mínimo implementado en **Java (con LangChain4j)** y **Python (con LangChain)**. Implementa las cuatro fases del módulo 5 del curso *IA generativa en el desarrollo de software*, utilizando un corpus de ejemplo compartido.

## Organización del Repositorio (Ramas)

Este proyecto está organizado en ramas para separar y facilitar el aprendizaje de cada tecnología de manera aislada:

*   **[`main`](https://github.com/leroidubuffet/mod5-rag-hello-world/tree/main)** (esta rama): Implementación en **Java Puro (POJO)**.
*   **[`python`](https://github.com/leroidubuffet/mod5-rag-hello-world/tree/python)**: Implementación en **Python 3 + LangChain**.
*   **[`spring-boot-migration`](https://github.com/leroidubuffet/mod5-rag-hello-world/tree/spring-boot-migration)**: Implementación en **Spring Boot 3 + LangChain4j**.

Cada rama contiene de forma exclusiva los archivos y la documentación correspondiente a su tecnología.

---

# Variante: Java Puro (POJO) — Rama `main`

Esta rama contiene la versión de Java Puro (POJO) utilizando LangChain4j. Todo se inicializa de forma manual y explícita, ideal para entender el flujo básico sin dependencias de frameworks complejos.

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

Primero, exporta tu clave de API en la terminal:
```bash
export ANTHROPIC_API_KEY=sk-ant-your-key-here
```

Además, puedes modificar las instrucciones del sistema editando el archivo [system_prompt.txt](./system_prompt.txt) en la raíz del repositorio.

### 1. Ejecutar Modo Lote (Batch Questions)
```bash
mvn clean compile exec:java
```

### 2. Ejecutar Modo Chat Interactivo
```bash
mvn clean compile exec:java -Dexec.mainClass="com.example.rag.HelloRagChat"
```
Escribe `salir`, `exit` o `quit` en la terminal para cerrar la sesión.

---

## Flujo del RAG (4 Fases)
1. **Fase 1: Indexación (al arrancar)**: Carga los documentos markdown de la carpeta `/corpus`, los divide en chunks de 300 caracteres (overlap de 30), genera embeddings locales con el modelo BGE-small-en-v1.5 y los guarda en un store en memoria.
2. **Fase 2: Recuperación (por pregunta)**: Genera el embedding de la pregunta y busca los top-3 fragmentos más similares.
3. **Fase 3: Ensamblado**: Agrupa los fragmentos recuperados con la pregunta en un prompt delimitado por etiquetas XML.
4. **Fase 4: Generación**: Envía el prompt a Claude Haiku y retorna la respuesta final justificando la fuente.
