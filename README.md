# mod5-rag-hello-world (Rama Python)

Pipeline RAG mínimo implementado en **Python (con LangChain)**. Implementa las cuatro fases del módulo 5 del curso *IA generativa en el desarrollo de software*, utilizando un corpus de ejemplo.

Esta rama (`python`) contiene exclusivamente la implementación de Python.

---

## Requisitos

- **API key de Anthropic**: Necesitas una clave con acceso a Claude Haiku.
- **Python 3.10+**

---

## Configuración y Ejecución

Primero, exporta tu clave de API en la terminal:
```bash
export ANTHROPIC_API_KEY=sk-ant-your-key-here
```

Además, puedes modificar las instrucciones del sistema editando el archivo [system_prompt.txt](./system_prompt.txt) en la raíz.

Entra en la carpeta de Python:
```bash
cd python
```

### 1. Preparar Entorno Virtual (solo la primera vez)
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Modo Lote (Preguntas prefijadas)
Este script ejecuta automáticamente tres preguntas predefinidas y muestra las respuestas generadas, citando las fuentes.
```bash
python hello_rag.py
```

### 3. Modo Chat Interactivo
Inicia un bucle de chat interactivo en la terminal con memoria conversacional y condensación de consultas.
```bash
python hello_rag_chat.py
```
Escribe `salir`, `exit` o `quit` para cerrar el chat.

---

## Variante con ChromaDB

Por defecto los vectores se almacenan en memoria y no persisten entre ejecuciones. Para usar ChromaDB con persistencia real:

**1. Levantar ChromaDB:**

```bash
docker compose up -d
```

**2. Verificar que responde:**

```bash
curl http://localhost:8000/api/v2/heartbeat   # debe devolver {"nanosecond heartbeat": ...}
```

Los scripts `hello_rag.py` y `hello_rag_chat.py` ya están configurados para conectarse a ChromaDB en `localhost:8000` y usar la colección `casa-tortuga`. Los vectores persisten entre ejecuciones y se pueden inspeccionar con:

```python
import chromadb
c = chromadb.HttpClient(host="localhost", port=8000)
print(c.get_collection("casa-tortuga").count())
```

Para detener ChromaDB conservando los datos:
```bash
docker compose down
```

Para detener y borrar el volumen:
```bash
docker compose down -v
```

---

## Estructura de la rama

- [`corpus/`](./corpus): Documentación de la tienda de ejemplo "Casa Tortuga".
- [`python/`](./python): Código fuente en Python.
- [`system_prompt.txt`](./system_prompt.txt): Archivo de prompt de sistema externalizado.
