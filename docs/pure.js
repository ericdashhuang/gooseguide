// pure, dom free helpers used by app.js, kept in their own module (no
// document/localStorage/fetch access) so they can be unit tested directly
// with node's built in test runner, with nothing to mock out

export function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

export function buildSystemPrompt() {
  return "You are a helpful assistant answering questions about being a first-year student at the University of Waterloo, based only on the provided context from uwaterloo.ca. If the context doesn't contain the answer, say so plainly instead of guessing. Write in plain prose. Do not use any citation markup like [1] or 【source】 - instead name the source title(s) in a sentence at the end of your answer.";
}

export function escapeHtml(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

// strips stray tool citation artifacts some models emit (eg 【source†l4-l9】)
// and renders basic **bold** markdown, since the answer is plain text from the model
// html escapes first, so this is safe to feed straight to innerHTML even though
// the input ultimately traces back to scraped web content the model saw
export function formatAnswer(raw) {
  const cleaned = raw
    .replace(/【[^】]*】/g, "")
    .replace(/[ \t]{2,}/g, " ")
    .trim();
  const escaped = escapeHtml(cleaned).replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  const paragraphs = escaped.split(/\n{2,}/).map((p) => `<p>${p.replace(/\n/g, "<br>")}</p>`);
  return paragraphs.join("");
}

export function buildContext(turnChunks) {
  return turnChunks
    .map(({ chunk }, i) => `[${i + 1}] Source: ${chunk.source_title} (${chunk.source_url})\n${chunk.text}`)
    .join("\n\n");
}

export function buildSummarySystemPrompt() {
  return "You maintain a short running summary of a conversation between a student and an assistant about being a first-year student at the University of Waterloo. Given the summary so far, the assistant's latest answer if there is one, and the student's new question, write an updated summary in 1-3 sentences (under 60 words) that captures the topic and every constraint the student has mentioned (program, residency status, co-op stream, and so on), including what the new question asks. Output only the summary.";
}

// user message for the summarizer call, the answer section is left out when
// the previous question was never answered (no generate click yet)
export function buildSummaryPrompt(prevSummary, lastAnswer, newQuestion) {
  const answerPart = lastAnswer ? `Assistant's latest answer:\n${lastAnswer}\n\n` : "";
  return `Summary so far:\n${prevSummary}\n\n${answerPart}New question: ${newQuestion}`;
}

// text that gets embedded for retrieval, the summary already carries every
// constraint from earlier turns, the question is repeated so its own wording
// still counts, with no summary (first turn, no api key, or a failed call) it
// falls back to the old behaviour of prepending just the previous question
export function buildRetrievalText(summary, prevQuestion, question, turnIndex) {
  if (summary) return `${summary}\n${question}`;
  if (turnIndex > 0 && prevQuestion) return `${prevQuestion}\n${question}`;
  return question;
}

// user message for the answer call, a short summary stands in for the full
// prior q/a transcript so the prompt stays the same size however long the chat is
export function buildUserContent(summary, context, question) {
  const summaryPart = summary ? `Conversation summary:\n${summary}\n\n` : "";
  return `${summaryPart}Context:\n\n${context}\n\nQuestion: ${question}`;
}
