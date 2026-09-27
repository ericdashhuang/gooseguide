import { test } from "node:test";
import assert from "node:assert/strict";
import { dot, escapeHtml, formatAnswer, buildContext, buildHistoryText } from "./pure.js";

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

test("buildHistoryText() includes only prior turns that already have an answer", () => {
  const turns = [
    { question: "How many co-op terms do I need?", answer: "Three." },
    { question: "What about abroad?", answer: null }, // no answer yet, should be excluded
  ];
  const history = buildHistoryText(turns, 2);
  assert.ok(history.includes("Q: How many co-op terms do I need?\nA: Three."));
  assert.ok(!history.includes("What about abroad?"));
});

test("buildHistoryText() excludes the current turn itself, not just future ones", () => {
  const turns = [
    { question: "First question", answer: "First answer" },
    { question: "Second question", answer: "Second answer" },
  ];
  // uptoTurnIndex=1 means "turns before index 1", ie only the first turn
  const history = buildHistoryText(turns, 1);
  assert.ok(history.includes("First question"));
  assert.ok(!history.includes("Second question"));
});

test("buildHistoryText() returns an empty string for the very first turn", () => {
  assert.equal(buildHistoryText([], 0), "");
});
