package com.example.rag;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingMatch;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import dev.langchain4j.store.embedding.EmbeddingStore;

import java.util.List;

/**
 * Fase 2 — RECUPERACION (por consulta):
 * Genera el embedding de la pregunta del usuario y busca los k chunks
 * mas cercanos en la base vectorial por similitud coseno.
 */
public class RetrievalService {

    private final EmbeddingModel embeddingModel;
    private final EmbeddingStore<TextSegment> store;

    public RetrievalService(EmbeddingModel embeddingModel, EmbeddingStore<TextSegment> store) {
        this.embeddingModel = embeddingModel;
        this.store = store;
    }

    public List<EmbeddingMatch<TextSegment>> retrieve(String question, int k, double minScore) {
        Embedding queryEmbedding = embeddingModel.embed(question).content();

        EmbeddingSearchRequest request = EmbeddingSearchRequest.builder()
                .queryEmbedding(queryEmbedding)
                .maxResults(k)
                .minScore(minScore)
                .build();

        return store.search(request).matches();
    }
}
