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

// prior turns that got an answer, formatted as plain Q/A text so the llm can
// refer back to what it already told you this conversation, takes the turns
// array explicitly (rather than closing over module state) so it's testable
// on its own
export function buildHistoryText(turns, uptoTurnIndex) {
  return turns
    .slice(0, uptoTurnIndex)
    .filter((t) => t.answer)
    .map((t) => `Q: ${t.question}\nA: ${t.answer}`)
    .join("\n\n");
}
