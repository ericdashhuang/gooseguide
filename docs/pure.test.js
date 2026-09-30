import { test } from "node:test";
import assert from "node:assert/strict";
import { dot, escapeHtml, formatAnswer, buildContext, buildSummaryPrompt, buildRetrievalText, buildUserContent } from "./pure.js";

test("dot() computes the dot product", () => {
  assert.equal(dot([1, 2, 3], [4, 5, 6]), 1 * 4 + 2 * 5 + 3 * 6);
});

test("dot() ranks a closer vector above a farther one", () => {
  // query, a near match, and a clearly unrelated one
  const query = [1, 0, 0];
  const near = [0.9, 0.1, 0];
  const far = [0, 0, 1];
  assert.ok(dot(query, near) > dot(query, far));
});

test("escapeHtml() neutralizes HTML special characters", () => {
  assert.equal(escapeHtml("<script>&"), "&lt;script&gt;&amp;");
});

test("escapeHtml() leaves plain text untouched", () => {
  assert.equal(escapeHtml("just plain text, nothing special"), "just plain text, nothing special");
});

test("formatAnswer() renders **bold** as <strong>", () => {
  const html = formatAnswer("You need **three work terms**.");
  assert.ok(html.includes("<strong>three work terms</strong>"));
});

test("formatAnswer() strips tool-citation artifacts like 【1†L4-L9】", () => {
  const html = formatAnswer("Three terms are required【1†L4-L9】【3†L9-L13】.");
  assert.ok(!html.includes("【"));
  assert.ok(!html.includes("】"));
});

test("formatAnswer() escapes raw HTML before adding its own tags (no XSS from model output)", () => {
  const html = formatAnswer("<img src=x onerror=alert(1)> **bold**");
  assert.ok(!html.includes("<img"));
  assert.ok(html.includes("&lt;img"));
  assert.ok(html.includes("<strong>bold</strong>"));
});

test("formatAnswer() collapses runs of spaces left over after stripping artifacts", () => {
  const html = formatAnswer("degree.  Your program may also allow flexible terms.");
  assert.ok(!html.includes("  "));
});

test("buildContext() numbers each chunk and includes its source", () => {
  const turnChunks = [
    { chunk: { source_title: "Work term requirements", source_url: "https://example.com/a", text: "Body A" } },
    { chunk: { source_title: "Co-op rules", source_url: "https://example.com/b", text: "Body B" } },
  ];
  const context = buildContext(turnChunks);
  assert.ok(context.includes("[1] Source: Work term requirements (https://example.com/a)\nBody A"));
  assert.ok(context.includes("[2] Source: Co-op rules (https://example.com/b)\nBody B"));
});

test("buildSummaryPrompt() includes the prior summary, the latest answer and the new question", () => {
  const prompt = buildSummaryPrompt(
    "Asked how many co-op terms are required.",
    "Most programs need five work terms.",
    "What about for international students?"
  );
  assert.ok(prompt.includes("Asked how many co-op terms are required."));
  assert.ok(prompt.includes("Most programs need five work terms."));
  assert.ok(prompt.includes("What about for international students?"));
});

test("buildSummaryPrompt() omits the answer section when the last question was never answered", () => {
  const prompt = buildSummaryPrompt("Asked about co-op terms.", null, "What about abroad?");
  assert.ok(!prompt.includes("latest answer"));
  assert.ok(prompt.includes("What about abroad?"));
});

test("buildRetrievalText() embeds the summary plus the new question, so constraints accumulate", () => {
  const text = buildRetrievalText(
    "International students asking how many co-op terms are required.",
    "How many co-op terms for international students?",
    "What about for environment majors?",
    2
  );
  assert.ok(text.includes("International students"));
  assert.ok(text.includes("What about for environment majors?"));
});

test("buildRetrievalText() falls back to the previous question when there is no summary", () => {
  const text = buildRetrievalText(null, "How many co-op terms do I need?", "What about abroad?", 1);
  assert.equal(text, "How many co-op terms do I need?\nWhat about abroad?");
});

test("buildRetrievalText() is just the question on the first turn", () => {
  assert.equal(buildRetrievalText(null, null, "How many co-op terms do I need?", 0), "How many co-op terms do I need?");
});

test("buildUserContent() includes the summary only when there is one", () => {
  const withSummary = buildUserContent("Topic: co-op terms.", "CTX", "Q?");
  assert.ok(withSummary.includes("Conversation summary:\nTopic: co-op terms."));
  assert.ok(withSummary.includes("Context:\n\nCTX"));
  assert.ok(withSummary.endsWith("Question: Q?"));
  assert.ok(!buildUserContent(null, "CTX", "Q?").includes("Conversation summary"));
});
