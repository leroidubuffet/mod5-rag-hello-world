package com.example.rag;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.anthropic.AnthropicChatModel;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.bgesmallzhv15q.BgeSmallZhV15QuantizedEmbeddingModel;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.store.embedding.EmbeddingMatch;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;
import java.util.Scanner;

/**
 * Interactive chat client for the RAG pipeline.
 */
public class HelloRagChat {

    public static void main(String[] args) {
        // 0. Cargar configuracion
        Properties props = loadProperties();
        int chunkSize = Integer.parseInt(props.getProperty("rag.chunk.size", "300"));
        int chunkOverlap = Integer.parseInt(props.getProperty("rag.chunk.overlap", "30"));
        int k = Integer.parseInt(props.getProperty("rag.retrieval.k", "3"));
        double minScore = Double.parseDouble(props.getProperty("rag.retrieval.min_score", "0.5"));
        String chatModelName = props.getProperty("rag.model.chat", "claude-haiku-4-5-20251001");

        // 1. Inicializar modelos y almacenamiento
        EmbeddingModel embeddingModel = new BgeSmallZhV15QuantizedEmbeddingModel();
        EmbeddingStore<TextSegment> store = new InMemoryEmbeddingStore<>();
        ChatModel chatModel = AnthropicChatModel.builder()
                .apiKey(requireEnv("ANTHROPIC_API_KEY"))
                .modelName(chatModelName)
                .temperature(0.0)
                .build();

        // 2. Ingesta e indexación de documentos
        IngestionService ingestionService = new IngestionService(embeddingModel, store);
        RetrievalService retrievalService = new RetrievalService(embeddingModel, store);

        System.out.println("Cargando y procesando corpus...");
        ingestionService.indexDocuments(Path.of("corpus"), chunkSize, chunkOverlap);
        System.out.println("¡Sistema RAG listo! Escribe 'salir' o 'exit' para terminar.\n");

        // 3. Inicializar historial y prompt de sistema
        List<ChatMessage> chatHistory = new ArrayList<>();
        String systemPrompt = loadSystemPrompt();
        String systemInstruction = "Eres un asistente virtual de atención al cliente de 'Casa Tortuga'.\n\n" + systemPrompt;

        // Bucle de conversacion
        Scanner scanner = new Scanner(System.in);
        while (true) {
            System.out.print("Tú: ");
            if (!scanner.hasNextLine()) {
                break;
            }
            String userInput = scanner.nextLine().trim();
            if (userInput.isEmpty()) {
                continue;
            }
            if (userInput.equalsIgnoreCase("salir") || userInput.equalsIgnoreCase("exit") || userInput.equalsIgnoreCase("quit")) {
                System.out.println("Asistente: ¡Hasta luego!");
                break;
            }

            try {
                // A. Condensar pregunta si hay historial
                String searchQuery = condenseQuery(chatHistory, userInput, chatModel);
                if (!searchQuery.equals(userInput)) {
                    System.out.printf("%n[Búsqueda RAG optimizada: '%s']%n", searchQuery);
                }

                // B. Recuperación de fragmentos
                List<EmbeddingMatch<TextSegment>> matches = retrievalService.retrieve(searchQuery, k, minScore);

                // C. Construir contexto
                String contextStr = buildContextStr(matches);

                // D. Construir lista de mensajes para el LLM
                List<ChatMessage> messages = new ArrayList<>();
                messages.add(new SystemMessage(systemInstruction));
                messages.addAll(chatHistory);

                String currentPrompt = "<context>\n" + contextStr + "</context>\n\n" +
                                       "Pregunta del usuario: " + userInput;
                messages.add(new UserMessage(currentPrompt));

                // E. Generación de respuesta
                System.out.print("Asistente: Pensando...\r");
                ChatResponse response = chatModel.chat(messages);
                String answer = response.aiMessage().text();
                System.out.println("Asistente: " + answer + "\n");

                // F. Actualizar historial de chat
                chatHistory.add(new UserMessage(userInput));
                chatHistory.add(new AiMessage(answer));

            } catch (Exception e) {
                System.out.println("\nError: " + e.getMessage() + "\n");
            }
        }
        scanner.close();
    }

    private static String condenseQuery(List<ChatMessage> chatHistory, String latestQuestion, ChatModel chatModel) {
        if (chatHistory.isEmpty()) {
            return latestQuestion;
        }

        StringBuilder historyBuilder = new StringBuilder();
        for (ChatMessage msg : chatHistory) {
            String role = (msg instanceof UserMessage) ? "Usuario" : "Asistente";
            String text = "";
            if (msg instanceof UserMessage userMsg) {
                text = userMsg.singleText();
            } else if (msg instanceof AiMessage aiMsg) {
                text = aiMsg.text();
            }
            if (text.contains("Pregunta del usuario:")) {
                text = text.substring(text.lastIndexOf("Pregunta del usuario:") + "Pregunta del usuario:".length()).trim();
            }
            historyBuilder.append(role).append(": ").append(text).append("\n");
        }

        String condensePrompt =
            "Dado el siguiente historial de conversación y una nueva pregunta del usuario, " +
            "crea una consulta de búsqueda independiente y optimizada en español para buscar en una base de datos vectorial.\n" +
            "La consulta debe ser concisa y resolver pronombres o referencias contextuales.\n\n" +
            "Historial de conversación:\n" + historyBuilder.toString() + "\n" +
            "Nueva pregunta: " + latestQuestion + "\n\n" +
            "Consulta de búsqueda optimizada (responde SOLO con la consulta):";

        String condensed = chatModel.chat(condensePrompt);
        if (condensed != null) {
            condensed = condensed.trim().replaceAll("^\"|\"$|^'|'$", "");
        }
        return condensed;
    }

    private static String buildContextStr(List<EmbeddingMatch<TextSegment>> matches) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < matches.size(); i++) {
            TextSegment segment = matches.get(i).embedded();
            String source = segment.metadata().getString("file_name");
            sb.append("<doc id=\"").append(i + 1).append("\"")
                    .append(" source=\"").append(source != null ? source : "desconocido").append("\">\n")
                    .append(segment.text()).append("\n")
                    .append("</doc>\n");
        }
        return sb.toString();
    }

    private static Properties loadProperties() {
        Properties props = new Properties();
        try (InputStream input = HelloRagChat.class.getClassLoader().getResourceAsStream("application.properties")) {
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

    private static String loadSystemPrompt() {
        Path path = Path.of("system_prompt.txt");
        if (!Files.exists(path)) {
            path = Path.of("../system_prompt.txt");
        }
        if (!Files.exists(path)) {
            path = Path.of("../../system_prompt.txt");
        }
        try {
            return Files.readString(path).strip();
        } catch (IOException e) {
            throw new RuntimeException("No se pudo leer el archivo system_prompt.txt en " + path.toAbsolutePath(), e);
        }
    }

    private static String requireEnv(String name) {
        String value = System.getenv(name);
        if (value == null || value.isBlank()) {
            throw new IllegalStateException("Variable de entorno '" + name + "' no definida.");
        }
        return value;
    }
}
