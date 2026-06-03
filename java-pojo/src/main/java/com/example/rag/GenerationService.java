package com.example.rag;

import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.store.embedding.EmbeddingMatch;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

/**
 * Fase 3 — ENSAMBLADO y Fase 4 — GENERACION.
 */
public class GenerationService {

    private static final Logger log = LoggerFactory.getLogger(GenerationService.class);

    private final ChatModel chatModel;
    private final boolean showPrompt;
    private final String systemInstruction;

    public GenerationService(ChatModel chatModel, boolean showPrompt, String systemInstruction) {
        this.chatModel = chatModel;
        this.showPrompt = showPrompt;
        this.systemInstruction = systemInstruction;
    }

    public String generateAnswer(String question, List<EmbeddingMatch<TextSegment>> matches) {
        String prompt = buildPrompt(matches, question);

        if (showPrompt) {
            System.out.println("\n------------------------------ PROMPT ENSAMBLADO ENVIADO AL MODELO ------------------------------");
            System.out.println(prompt);
            System.out.println("-------------------------------------------------------------------------------------------------");
        }

        return chatModel.chat(prompt);
    }

    private String buildPrompt(List<EmbeddingMatch<TextSegment>> matches, String question) {
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
                .append(systemInstruction)
                .append("\n</instructions>");

        return sb.toString();
    }
}
