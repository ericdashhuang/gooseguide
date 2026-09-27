import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2";
import { dot, buildSystemPrompt, formatAnswer, buildContext, buildHistoryText } from "./pure.js";

// let transformers.js fetch the model from the hf hub cdn (default) and cache
// it with the browser's cache api, so repeat visits skip the download
env.allowLocalModels = false;

const MODEL_ID = "Xenova/all-MiniLM-L6-v2";
const CACHE_KEY = "wfyr_embeddings_v1";
const TOP_K = 5;

const statusText = document.getElementById("statusText");
const statusRow = document.getElementById("status");
const progressBar = document.getElementById("progressBar");
const qform = document.getElementById("qform");
const questionInput = document.getElementById("question");
const askBtn = document.getElementById("askBtn");
const settingsCard = document.getElementById("settingsCard");
const conversationEl = document.getElementById("conversation");
const providerSelect = document.getElementById("provider");
const modelInput = document.getElementById("model");
const apiKeyInput = document.getElementById("apiKey");
const rememberKeyInput = document.getElementById("rememberKey");

let extractor = null;
let chunks = [];
let vectors = []; // parallel array of float32array, normalized

// each turn { question, chunks [{chunk, score}], answer string|null }
// kept around so a later question's retrieval and a later answer's prompt
// can both refer back to what was asked and answered before it
let turns = [];

function setStatus(text) {
  statusText.textContent = text;
}

async function embed(text) {
  const output = await extractor(text, { pooling: "mean", normalize: true });
  return Float32Array.from(output.data);
}

function loadCachedVectors(expectedCount) {
  try {
    const raw = localStorage.getItem(CACHE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed) || parsed.length !== expectedCount) return null;
    return parsed.map((arr) => Float32Array.from(arr));
  } catch {
    return null;
  }
}

function saveCachedVectors(vecs) {
  try {
    localStorage.setItem(CACHE_KEY, JSON.stringify(vecs.map((v) => Array.from(v))));
  } catch {
    // storage full or unavailable, fine, we just reembed next visit
  }
}

async function init() {
  setStatus("Loading embedding model (first visit only, ~25MB)...");
  extractor = await pipeline("feature-extraction", MODEL_ID, {
    progress_callback: (p) => {
      if (p.status === "progress" && p.total) {
        const pct = Math.round((p.loaded / p.total) * 100);
        setStatus(`Downloading embedding model... ${pct}%`);
      }
    },
  });

  const resp = await fetch("data/chunks.json");
  chunks = await resp.json();

  const cached = loadCachedVectors(chunks.length);
  if (cached) {
    vectors = cached;
    setStatus(`Ready - ${chunks.length} passages indexed (cached).`);
  } else {
    setStatus(`Embedding ${chunks.length} passages in your browser (one-time, ~30-60s)...`);
    progressBar.hidden = false;
    progressBar.max = chunks.length;
    vectors = [];
    for (let i = 0; i < chunks.length; i++) {
      vectors.push(await embed(chunks[i].text));
      progressBar.value = i + 1;
      if (i % 10 === 0) setStatus(`Embedding passages... ${i + 1}/${chunks.length}`);
    }
    progressBar.hidden = true;
    saveCachedVectors(vectors);
    setStatus(`Ready - ${chunks.length} passages indexed.`);
  }

  statusRow.querySelector(".spinner").remove();
  questionInput.disabled = false;
  askBtn.disabled = false;

  const rememberedKey = localStorage.getItem("wfyr_api_key");
  if (rememberedKey) {
    apiKeyInput.value = rememberedKey;
    rememberKeyInput.checked = true;
  }
}

async function callAnthropic(apiKey, model, userContent) {
  const resp = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: {
      "content-type": "application/json",
      "x-api-key": apiKey,
      "anthropic-version": "2023-06-01",
      "anthropic-dangerous-direct-browser-access": "true",
    },
    body: JSON.stringify({
      model,
      max_tokens: 600,
      system: buildSystemPrompt(),
      messages: [{ role: "user", content: userContent }],
    }),
  });
  if (!resp.ok) throw new Error(`Anthropic API error ${resp.status}: ${await resp.text()}`);
  const data = await resp.json();
  return data.content[0].text;
}

async function callGroq(apiKey, model, userContent) {
  const resp = await fetch("https://api.groq.com/openai/v1/chat/completions", {
    method: "POST",
    headers: {
      "content-type": "application/json",
      authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model,
      messages: [
        { role: "system", content: buildSystemPrompt() },
        { role: "user", content: userContent },
      ],
    }),
  });
  if (!resp.ok) throw new Error(`Groq API error ${resp.status}: ${await resp.text()}`);
  const data = await resp.json();
  return data.choices[0].message.content;
}

const PROVIDER_CALLS = {
  anthropic: callAnthropic,
  groq: callGroq,
};

const DEFAULT_MODELS = {
  anthropic: "claude-sonnet-5",
  groq: "openai/gpt-oss-120b",
};

function syncModelField() {
  modelInput.value = DEFAULT_MODELS[providerSelect.value] || "";
}

providerSelect.addEventListener("change", syncModelField);
// some browsers restore a <select>'s value on reload/back-forward without
// firing "change", which would leave the model field out of sync, so also
// sync once up front against whatever the provider field actually shows
syncModelField();

function wireGenerateButton(turnIndex, question, turnChunks, generateBtn, errorEl, answerEl) {
  generateBtn.addEventListener("click", async () => {
    const apiKey = apiKeyInput.value.trim();
    const model = modelInput.value.trim();
    errorEl.hidden = true;
    answerEl.textContent = "";

    if (!apiKey) {
      errorEl.textContent = "Enter an API key first, then click Generate answer.";
      errorEl.hidden = false;
      return;
    }

    if (rememberKeyInput.checked) {
      try { localStorage.setItem("wfyr_api_key", apiKey); } catch { /* ignore */ }
    } else {
      try { localStorage.removeItem("wfyr_api_key"); } catch { /* ignore */ }
    }

    generateBtn.disabled = true;
    generateBtn.textContent = "Generating...";
    try {
      const history = buildHistoryText(turns, turnIndex);
      const context = buildContext(turnChunks);
      const userContent = `${history ? `Conversation so far:\n${history}\n\n` : ""}Context:\n\n${context}\n\nQuestion: ${question}`;
      const call = PROVIDER_CALLS[providerSelect.value];
      const answer = await call(apiKey, model, userContent);
      turns[turnIndex].answer = answer;
      answerEl.innerHTML = formatAnswer(answer);
    } catch (err) {
      errorEl.textContent = `${err.message} (if this looks like a CORS/network error, the provider may not allow direct browser calls -- try the other provider, or run rag/query.py locally instead).`;
      errorEl.hidden = false;
    } finally {
      generateBtn.disabled = false;
      generateBtn.textContent = "Generate answer";
    }
  });
}

function renderTurn(question, turnChunks, turnIndex) {
  const card = document.createElement("div");
  card.className = "card turn";

  const qEl = document.createElement("div");
  qEl.className = "turn-question";
  qEl.textContent = question;
  card.appendChild(qEl);

  const details = document.createElement("details");
  details.className = "turn-chunks";
  const summary = document.createElement("summary");
  summary.textContent = `${turnChunks.length} retrieved passages`;
  details.appendChild(summary);

  for (const { chunk, score } of turnChunks) {
    const div = document.createElement("div");
    div.className = "chunk";
    div.innerHTML = `
      <a class="chunk-source" href="${chunk.source_url}" target="_blank" rel="noopener">${chunk.source_title || chunk.source_url}</a>
      <span class="hint"> (similarity ${score.toFixed(2)})</span>
      <div class="chunk-text"></div>
    `;
    div.querySelector(".chunk-text").textContent = chunk.text;
    details.appendChild(div);
  }
  card.appendChild(details);

  const generateBtn = document.createElement("button");
  generateBtn.type = "button";
  generateBtn.textContent = "Generate answer";
  card.appendChild(generateBtn);

  const errorEl = document.createElement("div");
  errorEl.className = "error";
  errorEl.hidden = true;
  card.appendChild(errorEl);

  const answerEl = document.createElement("div");
  answerEl.className = "answer";
  card.appendChild(answerEl);

  conversationEl.appendChild(card);
  card.scrollIntoView({ behavior: "smooth", block: "start" });

  wireGenerateButton(turnIndex, question, turnChunks, generateBtn, errorEl, answerEl);
}

qform.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = questionInput.value.trim();
  if (!question || vectors.length === 0) return;

  askBtn.disabled = true;
  askBtn.textContent = "Searching...";

  const turnIndex = turns.length;
  // prepend the previous question (not its answer, keeps the embedding
  // input short) so a short follow up like "what about abroad?" retrieves
  // against the topic it's actually continuing, not just the fragment
  const retrievalText = turnIndex === 0 ? question : `${turns[turnIndex - 1].question}\n${question}`;

  const qVec = await embed(retrievalText);
  const scored = vectors.map((v, i) => ({ chunk: chunks[i], score: dot(qVec, v) }));
  scored.sort((a, b) => b.score - a.score);
  const topResults = scored.slice(0, TOP_K);

  renderTurn(question, topResults, turnIndex);
  turns.push({ question, chunks: topResults, answer: null });

  settingsCard.hidden = false;
  questionInput.value = "";
  questionInput.focus();
  askBtn.disabled = false;
  askBtn.textContent = "Ask";
});

init().catch((err) => {
  setStatus(`Failed to load: ${err.message}`);
  console.error(err);
});
