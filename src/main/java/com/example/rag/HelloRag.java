package com.example.rag;

import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.anthropic.AnthropicChatModel;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.bgesmallenv15.BgeSmallEnV15EmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingMatch;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Path;
import java.util.List;
import java.util.Properties;

/**
 * Orquestador del pipeline RAG refactorizado con configuracion externa.
 */
public class HelloRag {

    public static void main(String[] args) {
        // 0. Cargar configuracion
        Properties props = loadProperties();
        int chunkSize = Integer.parseInt(props.getProperty("rag.chunk.size", "300"));
        int chunkOverlap = Integer.parseInt(props.getProperty("rag.chunk.overlap", "30"));
        int k = Integer.parseInt(props.getProperty("rag.retrieval.k", "3"));
        double minScore = Double.parseDouble(props.getProperty("rag.retrieval.min_score", "0.5"));
        String chatModelName = props.getProperty("rag.model.chat", "claude-haiku-4-5-20251001");
        boolean showPrompt = Boolean.parseBoolean(props.getProperty("rag.debug.show_prompt", "false"));

        // 1. Inicializar modelos y almacenamiento
        EmbeddingModel embeddingModel = new BgeSmallEnV15EmbeddingModel();
        EmbeddingStore<TextSegment> store = new InMemoryEmbeddingStore<>();
        ChatModel chatModel = AnthropicChatModel.builder()
                .apiKey(requireEnv("ANTHROPIC_API_KEY"))
                .modelName(chatModelName)
                .build();

        // 2. Inicializar servicios
        IngestionService ingestionService = new IngestionService(embeddingModel, store);
        RetrievalService retrievalService = new RetrievalService(embeddingModel, store);
        GenerationService generationService = new GenerationService(chatModel, showPrompt);

        // 3. Fase 1: INDEXACION
        ingestionService.indexDocuments(Path.of("corpus"), chunkSize, chunkOverlap);

        // 4. Consultas
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
            List<EmbeddingMatch<TextSegment>> matches = retrievalService.retrieve(question, k, minScore);

            if (matches.isEmpty()) {
                System.out.printf("Sin fragmentos relevantes (score < %.1f).%n%n", minScore);
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

            // Fase 3 y 4: GENERACION
            String answer = generationService.generateAnswer(question, matches);
            System.out.printf("%nRESPUESTA:%n%s%n%n", answer);
        }
    }

    private static Properties loadProperties() {
        Properties props = new Properties();
        try (InputStream input = HelloRag.class.getClassLoader().getResourceAsStream("application.properties")) {
            if (input == null) {
                System.out.println("(!) No se encontro application.properties. Usando valores por defecto.");
                return props;
            }
            props.load(input);
        } catch (IOException ex) {
            System.err.println("(!) Error cargando application.properties: " + ex.getMessage());
        }
        return props;
    }

    private static String requireEnv(String name) {
        String value = System.getenv(name);
        if (value == null || value.isBlank()) {
            throw new IllegalStateException("Variable de entorno '" + name + "' no definida.");
        }
        return value;
    }
}
