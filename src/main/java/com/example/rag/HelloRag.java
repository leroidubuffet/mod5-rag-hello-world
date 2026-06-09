package com.example.rag;

import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.store.embedding.EmbeddingMatch;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;

import java.nio.file.Path;
import java.util.List;

@SpringBootApplication
@ConditionalOnProperty(name = "rag.mode", havingValue = "batch", matchIfMissing = true)
public class HelloRag implements CommandLineRunner {

    private final IngestionService ingestionService;
    private final RetrievalService retrievalService;
    private final GenerationService generationService;

    @Value("${rag.chunk.size:300}")
    private int chunkSize;

    @Value("${rag.chunk.overlap:30}")
    private int chunkOverlap;

    @Value("${rag.retrieval.k:3}")
    private int k;

    @Value("${rag.retrieval.min_score:0.5}")
    private double minScore;

    public HelloRag(IngestionService ingestionService, 
                    RetrievalService retrievalService, 
                    GenerationService generationService) {
        this.ingestionService = ingestionService;
        this.retrievalService = retrievalService;
        this.generationService = generationService;
    }

    public static void main(String[] args) {
        SpringApplication.run(HelloRag.class, args);
    }

    @Override
    public void run(String... args) {
        // 1. Fase 1: INDEXACION
        ingestionService.indexDocuments(Path.of("corpus"), chunkSize, chunkOverlap);

        // 2. Consultas
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
                String preview = m.embedded().text().replace("\n", " ");
                if (preview.length() > 70) preview = preview.substring(0, 67) + "...";
                System.out.printf("  [%d] score=%.3f  %s%n",
                        i + 1, m.score(), preview);
            }

            // Fase 3 y 4: GENERACION
            String answer = generationService.generateAnswer(question, matches);
            System.out.printf("%nRESPUESTA:%n%s%n%n", answer);
        }
    }
}
