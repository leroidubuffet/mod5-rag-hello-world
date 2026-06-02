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

import java.nio.file.Path;
import java.util.List;

/**
 * Fase 1 — INDEXACION (offline/batch):
 * Carga documentos del corpus, los parte en chunks, genera un embedding
 * por chunk y los almacena en la base vectorial.
 */
public class IngestionService {

    private final EmbeddingModel embeddingModel;
    private final EmbeddingStore<TextSegment> store;

    public IngestionService(EmbeddingModel embeddingModel, EmbeddingStore<TextSegment> store) {
        this.embeddingModel = embeddingModel;
        this.store = store;
    }

    public void indexDocuments(Path corpusDir, int chunkSize, int chunkOverlap) {
        // Cargar todos los documentos del directorio corpus/.
        List<Document> documents = FileSystemDocumentLoader.loadDocuments(
                corpusDir, new TextDocumentParser());
        System.out.printf("Cargados %d documentos del corpus%n", documents.size());

        // Splitter recursivo: respeta la estructura del documento.
        DocumentSplitter splitter = DocumentSplitters.recursive(chunkSize, chunkOverlap);

        List<TextSegment> segments = documents.stream()
                .flatMap(doc -> splitter.split(doc).stream())
                .toList();
        System.out.printf("Generados %d chunks (tamano=%d, overlap=%d)%n",
                segments.size(), chunkSize, chunkOverlap);

        // Generar embeddings para todos los chunks por lotes y cargarlos en el store.
        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();
        store.addAll(embeddings, segments);
        System.out.printf("Indexados %d vectores%n%n", embeddings.size());
    }
}
