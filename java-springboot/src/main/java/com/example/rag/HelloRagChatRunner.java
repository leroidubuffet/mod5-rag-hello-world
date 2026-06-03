package com.example.rag;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.store.embedding.EmbeddingMatch;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

@Component
@ConditionalOnProperty(name = "rag.mode", havingValue = "chat")
public class HelloRagChatRunner implements CommandLineRunner {

    private final IngestionService ingestionService;
    private final RetrievalService retrievalService;
    private final ChatLanguageModel chatModel;

    @Value("${rag.chunk.size:300}")
    private int chunkSize;

    @Value("${rag.chunk.overlap:30}")
    private int chunkOverlap;

    @Value("${rag.retrieval.k:3}")
    private int k;

    @Value("${rag.retrieval.min_score:0.5}")
    private double minScore;

    public HelloRagChatRunner(IngestionService ingestionService,
                              RetrievalService retrievalService,
                              ChatLanguageModel chatModel) {
        this.ingestionService = ingestionService;
        this.retrievalService = retrievalService;
        this.chatModel = chatModel;
    }

    @Override
    public void run(String... args) {
        System.out.println("Cargando y procesando corpus...");
        ingestionService.indexDocuments(Path.of("corpus"), chunkSize, chunkOverlap);
        System.out.println("¡Sistema RAG listo! Escribe 'salir' o 'exit' para terminar.\n");

        List<ChatMessage> chatHistory = new ArrayList<>();
        String systemPrompt = loadSystemPrompt();
        String systemInstruction = "Eres un asistente virtual de atención al cliente de 'Casa Tortuga'.\n\n" + systemPrompt;

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
                Response<AiMessage> response = chatModel.generate(messages);
                String answer = response.content().text();
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

    private String condenseQuery(List<ChatMessage> chatHistory, String latestQuestion, ChatLanguageModel chatModel) {
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

        Response<AiMessage> condensedResponse = chatModel.generate(new UserMessage(condensePrompt));
        String condensed = condensedResponse.content().text();
        if (condensed != null) {
            condensed = condensed.trim().replaceAll("^\"|\"$|^'|'$", "");
        }
        return condensed;
    }

    private String buildContextStr(List<EmbeddingMatch<TextSegment>> matches) {
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
}
