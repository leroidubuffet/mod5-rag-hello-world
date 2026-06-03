# Java Spring Boot RAG Pipeline

Esta carpeta contiene la implementación del pipeline RAG estructurada con Spring Boot 3 y LangChain4j. Hace uso de inyección de dependencias, beans de configuración de Spring y properties externalizados en `application.properties`.

## Requisitos

- JDK 21
- Maven 3.9+
- API Key de Anthropic (con acceso a Claude Haiku)

## Configuración

1. Exporta tu API key de Anthropic en la terminal:
   ```bash
   export ANTHROPIC_API_KEY=sk-ant-your-key-here
   ```

2. Configura los parámetros (tamaño de chunk, modelo de chat, etc.) en `src/main/resources/application.properties`.

## Ejecución

Los dos modos se gestionan mediante perfiles/parámetros de configuración de Spring (`rag.mode`), evitando la necesidad de múltiples puntos de entrada de aplicación `@SpringBootApplication`.

### 1. Modo Lote (Batch Questions)
Ejecuta la secuencia de preguntas predeterminada sobre el corpus usando el runner `HelloRagRunner` (activo por defecto).

```bash
mvn clean compile spring-boot:run
```

O forzando explícitamente el modo batch:
```bash
mvn clean compile spring-boot:run -Dspring-boot.run.arguments="--rag.mode=batch"
```

### 2. Modo Chat Interactivo
Activa el runner `HelloRagChatRunner` para iniciar un bucle conversacional en terminal con memoria conversacional y condensación de consulta.

```bash
mvn clean compile spring-boot:run -Dspring-boot.run.arguments="--rag.mode=chat"
```

Escribe `salir`, `exit` o `quit` para cerrar el chat.
