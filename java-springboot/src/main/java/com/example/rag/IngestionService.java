package com.example.rag;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.TextDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingStore;
import org.springframework.stereotype.Service;

import java.nio.file.Path;
import java.util.List;

@Service
public class IngestionService {

    private final EmbeddingModel embeddingModel;
    private final EmbeddingStore<TextSegment> store;

    public IngestionService(EmbeddingModel embeddingModel, EmbeddingStore<TextSegment> store) {
        this.embeddingModel = embeddingModel;
        this.store = store;
    }

    public void indexDocuments(Path corpusDir, int chunkSize, int chunkOverlap) {
        List<Document> documents = FileSystemDocumentLoader.loadDocuments(
                corpusDir, new TextDocumentParser());
        System.out.printf("Cargados %d documentos del corpus%n", documents.size());

        DocumentSplitter splitter = DocumentSplitters.recursive(chunkSize, chunkOverlap);

        List<TextSegment> segments = documents.stream()
                .flatMap(doc -> splitter.split(doc).stream())
                .toList();
        System.out.printf("Generados %d chunks (tamano=%d, overlap=%d)%n",
                segments.size(), chunkSize, chunkOverlap);

        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();
        store.addAll(embeddings, segments);
        System.out.printf("Indexados %d vectores%n%n", embeddings.size());
    }
}
