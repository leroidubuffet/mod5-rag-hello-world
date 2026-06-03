# Java POJO RAG Pipeline

Esta carpeta contiene la versión pura de Java (POJO) para el pipeline RAG usando LangChain4j y Maven. Todo se inicializa de forma manual y explícita, ideal para aprender el flujo básico sin dependencias de frameworks complejos.

## Requisitos

- JDK 21
- Maven 3.9+
- API Key de Anthropic (con acceso a Claude Haiku)

## Configuración

1. Exporta tu API key de Anthropic en la terminal:
   ```bash
   export ANTHROPIC_API_KEY=sk-ant-your-key-here
   ```

2. Configura los parámetros adicionales (tamaño de chunk, overlap, etc.) en `src/main/resources/application.properties` si es necesario.

## Ejecución

### 1. Modo Lote (Batch Questions)
Ejecuta la clase principal predeterminada `HelloRag`, la cual procesa tres preguntas fijadas de antemano mostrando los fragmentos recuperados y la respuesta final con citas.

```bash
mvn clean compile exec:java
```

### 2. Modo Chat Interactivo
Ejecuta la clase de chat interactivo `HelloRagChat` que inicia un bucle conversacional en terminal. Utiliza memoria de sesión y realiza condensación de consultas mediante LLM antes de buscar en la base de datos de vectores en memoria.

```bash
mvn clean compile exec:java -Dexec.mainClass="com.example.rag.HelloRagChat"
```

Escribe `salir`, `exit` o `quit` para cerrar el chat.
