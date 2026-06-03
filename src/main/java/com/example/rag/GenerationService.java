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

    public GenerationService(ChatModel chatModel, boolean showPrompt) {
        this.chatModel = chatModel;
        this.showPrompt = showPrompt;
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
                .append("Responde a la pregunta planteada basándote ÚNICAMENTE en la información explícita que se encuentra dentro de las etiquetas <context>.\n\n")
                .append("Si la respuesta no se puede deducir directamente del contexto proporcionado, debes responder estrictamente con la siguiente frase literal: \"No tengo esa información en el corpus.\"\n\n")
                .append("Es obligatorio que justifiques tu respuesta citando textualmente el id y la fuente del documento que sustenta tu afirmación (por ejemplo: [doc id=\"1\" source=\"archivo.md\"]). Sé conciso y directo en tu respuesta.\n")
                .append("</instructions>");

        return sb.toString();
    }
}
