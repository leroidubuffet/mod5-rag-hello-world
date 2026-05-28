package com.example.rag;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.TextDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.anthropic.AnthropicChatModel;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.bgesmallenv15.BgeSmallEnV15EmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingMatch;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.nio.file.Path;
import java.util.List;

/**
 * Pipeline RAG minimo en una sola clase.
 *
 * Implementa las cuatro fases descritas en el modulo 5 del curso
 * "IA generativa en el desarrollo de software":
 *
 *   Fase 1 — INDEXACION (offline/batch):
 *     Carga documentos del corpus, los parte en chunks, genera un embedding
 *     por chunk y los almacena en la base vectorial.
 *
 *   Fase 2 — RECUPERACION (por consulta):
 *     Genera el embedding de la pregunta del usuario y busca los k chunks
 *     mas cercanos en la base vectorial por similitud coseno.
 *
 *   Fase 3 — ENSAMBLADO:
 *     Construye el prompt combinando los fragmentos recuperados (en bloques
 *     XML con id y fuente) y la pregunta, siguiendo el patron de la seccion
 *     5.4 del modulo.
 *
 *   Fase 4 — GENERACION:
 *     Llama al LLM con el prompt ensamblado y devuelve la respuesta.
 *
 * Requiere: JDK 21, Maven 3.9+, variable de entorno ANTHROPIC_API_KEY.
 * Ejecucion: mvn -q exec:java
 */
public class HelloRag {

    // -----------------------------------------------------------------------
    //  Parametros del pipeline.
    //  Modifica estos valores y re-ejecuta para observar el efecto (ver README).
    // -----------------------------------------------------------------------

    /** Tokens por chunk. Valores recomendados: 200-500. */
    private static final int CHUNK_SIZE = 300;

    /** Solapamiento entre chunks contiguos (~10% del tamano). */
    private static final int CHUNK_OVERLAP = 30;

    /** Numero de fragmentos a recuperar por consulta (top-k). */
    private static final int K = 3;

    /** Umbral de similitud coseno [0.0 - 1.0]. Fragmentos por debajo se descartan. */
    private static final double MIN_SCORE = 0.5;

    /** Modelo de chat de Anthropic. */
    private static final String CHAT_MODEL = "claude-haiku-4-5-20251001";

    public static void main(String[] args) {

        // -----------------------------------------------------------------------
        //  Componentes del pipeline
        // -----------------------------------------------------------------------

        // Modelo de embeddings local (in-process, sin API externa).
        // BgeSmallEnV15 empaqueta el modelo ONNX como recurso del JAR;
        // no requiere conexion a red mas alla de la descarga inicial de Maven.
        EmbeddingModel embeddingModel = new BgeSmallEnV15EmbeddingModel();

        // Store en memoria. Los vectores no persisten entre ejecuciones.
        // Para persistencia real, sustituir por ChromaEmbeddingStore
        // (ver seccion "Variante con ChromaDB" del README).
        EmbeddingStore<TextSegment> store = new InMemoryEmbeddingStore<>();

        // Modelo de chat de Anthropic. La API key se lee del entorno.
        ChatModel chatModel = AnthropicChatModel.builder()
                .apiKey(requireEnv("ANTHROPIC_API_KEY"))
                .modelName(CHAT_MODEL)
                .build();

        // -----------------------------------------------------------------------
        //  Fase 1: INDEXACION
        // -----------------------------------------------------------------------

        // Cargar todos los documentos del directorio corpus/.
        // FileSystemDocumentLoader lee cada fichero como un Document; el parser
        // de texto plano preserva el Markdown sin interpretarlo.
        Path corpusDir = Path.of("corpus");
        List<Document> documents = FileSystemDocumentLoader.loadDocuments(
                corpusDir, new TextDocumentParser());
        System.out.printf("Cargados %d documentos del corpus%n", documents.size());

        // Splitter recursivo: respeta la estructura del documento (secciones,
        // parrafos, frases) y solo cae a nivel de tokens cuando es necesario.
        // Esto produce chunks mas coherentes que un corte por tamano fijo.
        DocumentSplitter splitter = DocumentSplitters.recursive(CHUNK_SIZE, CHUNK_OVERLAP);

        List<TextSegment> segments = documents.stream()
                .flatMap(doc -> splitter.split(doc).stream())
                .toList();
        System.out.printf("Generados %d chunks (tamano=%d, overlap=%d)%n",
                segments.size(), CHUNK_SIZE, CHUNK_OVERLAP);

        // Generar embeddings para todos los chunks en una sola llamada por lotes
        // y cargarlos en el store junto con su metadata (texto original + fuente).
        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();
        store.addAll(embeddings, segments);
        System.out.printf("Indexados %d vectores%n%n", embeddings.size());

        // -----------------------------------------------------------------------
        //  Consultas de demostracion
        // -----------------------------------------------------------------------

        // Tres preguntas disenadas para ilustrar casos distintos:
        //   1. Respuesta directa en un solo documento.
        //   2. Respuesta en otro documento.
        //   3. Pregunta capciosa que requiere combinar dos documentos
        //      (el reembolso existe pero no aplica a Canarias).
        List<String> questions = List.of(
                "Cuanto cuesta enviar a Madrid?",
                "Cuantos dias tengo para devolver un producto?",
                "Aceptan pago contra reembolso a Tenerife?"
        );

        for (String question : questions) {
            System.out.println("=".repeat(70));
            System.out.println("PREGUNTA: " + question);
            System.out.println("=".repeat(70));

            // Fase 2: RECUPERACION
            List<EmbeddingMatch<TextSegment>> matches = retrieve(
                    question, embeddingModel, store);

            if (matches.isEmpty()) {
                System.out.printf(
                        "Sin fragmentos relevantes (score < %.1f). Respuesta directa.%n%n",
                        MIN_SCORE);
                continue;
            }

            System.out.println("Fragmentos recuperados:");
            for (int i = 0; i < matches.size(); i++) {
                EmbeddingMatch<TextSegment> m = matches.get(i);
                String source = m.embedded().metadata().getString("file_name");
                String preview = m.embedded().text().replace("\n", " ");
                if (preview.length() > 70) preview = preview.substring(0, 67) + "...";
                System.out.printf("  [%d] score=%.3f  src=%-18s  %s%n",
                        i + 1, m.score(), source, preview);
            }

            // Fase 3: ENSAMBLADO
            String prompt = buildPrompt(matches, question);

            // Fase 4: GENERACION
            String answer = chatModel.chat(prompt);
            System.out.printf("%nRESPUESTA:%n%s%n%n", answer);
        }
    }

    // -----------------------------------------------------------------------
    //  Fase 2: RECUPERACION
    // -----------------------------------------------------------------------

    /**
     * Embebe la pregunta con el mismo modelo usado en la indexacion y busca
     * los k fragmentos mas similares en el store, filtrando por MIN_SCORE.
     *
     * Es critico usar el mismo modelo en indexacion y recuperacion: cada
     * modelo define su propio espacio vectorial; mezclarlos produce
     * resultados sin sentido (ver seccion 5.5 del modulo).
     */
    private static List<EmbeddingMatch<TextSegment>> retrieve(
            String question,
            EmbeddingModel embeddingModel,
            EmbeddingStore<TextSegment> store) {

        Embedding queryEmbedding = embeddingModel.embed(question).content();

        EmbeddingSearchRequest request = EmbeddingSearchRequest.builder()
                .queryEmbedding(queryEmbedding)
                .maxResults(K)
                .minScore(MIN_SCORE)
                .build();

        return store.search(request).matches();
    }

    // -----------------------------------------------------------------------
    //  Fase 3: ENSAMBLADO DEL PROMPT
    // -----------------------------------------------------------------------

    /**
     * Construye el prompt siguiendo el patron de bloques XML de la seccion 5.4:
     *
     *   - Cada fragmento va en <doc id="N" source="fichero.md">.
     *     El id permite citar la fuente en la respuesta.
     *   - Las instrucciones van al final del prompt (efecto lost-in-the-middle:
     *     los modelos prestan mas atencion al principio y al final del contexto).
     *   - La instruccion explicita que hacer cuando la respuesta no esta en el
     *     corpus previene alucinaciones.
     */
    private static String buildPrompt(
            List<EmbeddingMatch<TextSegment>> matches,
            String question) {

        StringBuilder sb = new StringBuilder();

        sb.append("<context>\n");
        for (int i = 0; i < matches.size(); i++) {
            TextSegment segment = matches.get(i).embedded();
            String source = segment.metadata().getString("file_name");
            sb.append("<doc id=\"").append(i + 1).append("\"")
              .append(" source=\"").append(source != null ? source : "desconocido").append("\">\n")
              .append(segment.text()).append("\n")
              .append("</doc>\n");
        }
        sb.append("</context>\n\n");

        sb.append("<question>\n").append(question).append("\n</question>\n\n");

        sb.append("<instructions>\n")
          .append("Responde la pregunta basandote UNICAMENTE en los documentos del bloque\n")
          .append("context. Si la respuesta no esta alli, responde literalmente:\n")
          .append("\"No tengo esa informacion en el corpus.\"\n")
          .append("Cita el id del documento que sustenta tu afirmacion.\n")
          .append("Se conciso: maximo dos frases.\n")
          .append("</instructions>");

        return sb.toString();
    }

    // -----------------------------------------------------------------------
    //  Utilidades
    // -----------------------------------------------------------------------

    /**
     * Lee una variable de entorno obligatoria.
     * Lanza IllegalStateException si no esta definida o esta vacia,
     * con un mensaje de error accionable para el desarrollador.
     */
    private static String requireEnv(String name) {
        String value = System.getenv(name);
        if (value == null || value.isBlank()) {
            throw new IllegalStateException(
                    "Variable de entorno '" + name + "' no definida. " +
                    "Ejecuta: export " + name + "=<tu-clave>");
        }
        return value;
    }
}
