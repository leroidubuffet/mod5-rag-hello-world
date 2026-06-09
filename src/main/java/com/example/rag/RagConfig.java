package com.example.rag;

import dev.langchain4j.model.anthropic.AnthropicChatModel;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.bgesmallenv15q.BgeSmallEnV15QuantizedEmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.chroma.ChromaEmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import dev.langchain4j.data.segment.TextSegment;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;

@Configuration
public class RagConfig {

    @Value("${rag.model.chat}")
    private String modelName;

    @Value("${rag.store.type:memory}")
    private String storeType;

    @Value("${rag.chroma.url:http://localhost:8000}")
    private String chromaUrl;

    @Value("${rag.chroma.collection:casa-tortuga}")
    private String chromaCollection;

    @Bean
    public ChatLanguageModel chatModel() {
        String apiKey = System.getenv("ANTHROPIC_API_KEY");
        if (apiKey == null || apiKey.isBlank()) {
            throw new IllegalStateException("Variable de entorno 'ANTHROPIC_API_KEY' no definida.");
        }
        return AnthropicChatModel.builder()
                .apiKey(apiKey)
                .modelName(modelName)
                .build();
    }

    @Bean
    public EmbeddingModel embeddingModel() {
        return new BgeSmallEnV15QuantizedEmbeddingModel();
    }

    @Bean
    public EmbeddingStore<TextSegment> embeddingStore() {
        if ("chroma".equalsIgnoreCase(storeType)) {
            deleteChromaCollection();
            return ChromaEmbeddingStore.builder()
                    .baseUrl(chromaUrl)
                    .collectionName(chromaCollection)
                    .build();
        }
        return new InMemoryEmbeddingStore<>();
    }

    private void deleteChromaCollection() {
        try {
            HttpClient client = HttpClient.newHttpClient();
            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(chromaUrl + "/api/v1/collections/" + chromaCollection))
                    .DELETE()
                    .build();
            client.send(request, HttpResponse.BodyHandlers.discarding());
        } catch (Exception e) {
            // Collection may not exist yet — safe to ignore
        }
    }
}
