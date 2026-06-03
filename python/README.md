# Python RAG Pipeline

Esta carpeta contiene la implementación del pipeline RAG utilizando Python 3 y LangChain.

## Requisitos

- Python 3.10+
- Virtualenv (`python3 -m venv`)
- API Key de Anthropic (con acceso a Claude Haiku)

## Configuración

1. Crea y activa el entorno virtual:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Instala las dependencias necesarias:
   ```bash
   pip install -r requirements.txt
   ```

3. Exporta tu API key de Anthropic en la terminal:
   ```bash
   export ANTHROPIC_API_KEY=sk-ant-your-key-here
   ```

## Ejecución

### 1. Modo Lote (Batch Questions)
Este script ejecuta automáticamente tres preguntas predefinidas sobre el corpus en la carpeta raíz y muestra las respuestas generadas por Claude, citando sus fuentes.

```bash
python hello_rag.py
```

### 2. Modo Chat Interactivo
Este script inicia un bucle de chat interactivo (REPL) en la terminal. Cuenta con memoria conversacional e implementa la condensación de consultas (une el historial con tu nueva pregunta para optimizar la búsqueda vectorial).

```bash
python hello_rag_chat.py
```

Para salir del chat, escribe `salir`, `exit` o `quit`.
