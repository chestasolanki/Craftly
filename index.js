import dotenv from "dotenv";
dotenv.config();

import Groq from "groq-sdk";
import express from "express";
import { createServer } from "http";
import { spawn } from "child_process";
import path from "path";
import fs from "fs";
import net from "net";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const httpServer = createServer(app);
app.use(express.json());
app.use(express.static(__dirname));

const api = new Groq({
  apiKey: process.env.API_KEY
});

// ─── Per-session project dir ──────────────────────────────────────────────────
const projectDirs = new Map();

// ─── Free port finder ─────────────────────────────────────────────────────────
function findFreePort(base = 4000) {
  return new Promise((resolve) => {
    let port = base;
    const tryPort = () => {
      const s = net.createServer();
      s.once("error", () => { port++; tryPort(); });
      s.once("listening", () => { s.close(() => resolve(port)); });
      s.listen(port);
    };
    tryPort();
  });
}

// ─── Open browser ─────────────────────────────────────────────────────────────
function openBrowser(url) {
  const cmd =
    process.platform === "darwin" ? "open"
    : process.platform === "win32" ? "start"
    : "xdg-open";
  spawn(cmd, [url], { detached: true, stdio: "ignore" }).unref();
}

// ─── Tool 1: createFolder ─────────────────────────────────────────────────────
async function createFolder(name, sessionId) {
  try {
    const fullPath = path.join(__dirname, "projects", `${name}-${sessionId}`);
    fs.mkdirSync(fullPath, { recursive: true });
    projectDirs.set(sessionId, fullPath);
    return `Folder created at ${fullPath}`;
  } catch (err) {
    return `Error creating folder: ${err.message}`;
  }
}

// ─── Tool 2: writeFile ────────────────────────────────────────────────────────
async function writeFile(filename, content, sessionId) {
  try {
    const dir = projectDirs.get(sessionId);
    if (!dir) return "Error: No project folder. Call createFolder first.";
    const filePath = path.join(dir, filename);
    fs.writeFileSync(filePath, content, "utf8");
    return `File '${filename}' written (${content.length} chars)`;
  } catch (err) {
    return `Error writing file: ${err.message}`;
  }
}

// ─── Groq tools ───────────────────────────────────────────────────────────────
const TOOLS = [
  {
    type: "function",
    function: {
      name: "createFolder",
      description:
        "Creates the project folder. ALWAYS call this first before writing any files.",
      parameters: {
        type: "object",
        properties: {
          name: {
            type: "string",
            description:
              "Short, lowercase, hyphenated folder name (e.g. 'portfolio-site')",
          },
        },
        required: ["name"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "writeFile",
      description:
        "Writes a file inside the project folder. Use for index.html, style.css, script.js. Write COMPLETE, detailed, production-quality code.",
      parameters: {
        type: "object",
        properties: {
          filename: {
            type: "string",
            description: "Filename e.g. 'index.html', 'style.css', 'script.js'",
          },
          content: {
            type: "string",
            description:
              "Full file content. Must be complete and functional — no placeholders.",
          },
        },
        required: ["filename", "content"],
      },
    },
  },
];

// ─── Launch preview server ────────────────────────────────────────────────────
async function launchProject(sessionId, onEvent) {
  const projectPath = projectDirs.get(sessionId);
  if (!projectPath) {
    onEvent("status", "No project folder found — skipping launch.");
    return;
  }
  const indexFile = path.join(projectPath, "index.html");
  if (!fs.existsSync(indexFile)) {
    onEvent("status", "No index.html found — skipping launch.");
    return;
  }

  const port = await findFreePort(4000);

  // FIX: Wrap in Promise so we wait for the server to be READY before
  // calling openBrowser and emitting the launch event.
  await new Promise((resolve) => {
    const preview = express();
    // Explicitly serve index.html at root — fixes blank page on port 4000
    preview.get("/", (_req, res) => res.sendFile(indexFile));
    preview.use(express.static(projectPath));
    createServer(preview).listen(port, () => {
      const url = `http://localhost:${port}`;
      onEvent("launch", url);
      openBrowser(url);
      console.log(`Preview ready at ${url}`);
      resolve();
    });
  });
}

// ─── System prompt ────────────────────────────────────────────────────────────
const SYSTEM_PROMPT = `You are an expert web developer agent. Your job is to build complete, beautiful websites using your tools.

STRICT SEQUENCE — follow this every single time, no exceptions:
1. Call createFolder with a short hyphenated name like "portfolio-site".
2. Call writeFile for "index.html" — full, complete HTML with all content.
3. Call writeFile for "style.css" — rich, modern, detailed CSS with animations.
4. Call writeFile for "script.js" — interactive JavaScript.

QUALITY RULES:
- Write COMPLETE code for every file. No placeholders, no "add content here", no truncation.
- index.html must have real content: headings, paragraphs, cards, images (use https://picsum.photos for images), etc.
- style.css must include: Google Fonts import, CSS variables, responsive layout, hover effects, animations.
- script.js must add real interactivity relevant to the site.
- Each file must be substantial and production-quality.

Only make tool calls. Do not write any explanatory text.`;

// ─── Agent loop ───────────────────────────────────────────────────────────────
async function runAgentTurn(history, sessionId, onEvent) {
  let active = true;
  let iterations = 0;
  const MAX_ITERATIONS = 10;

  while (active && iterations < MAX_ITERATIONS) {
    iterations++;
    onEvent("status", "Agent is thinking...");

    const response = await api.chat.completions.create({
      model: "moonshotai/kimi-k2-instruct",
      messages: history,
      tools: TOOLS,
      tool_choice: "auto",
      // FIX: parallel_tool_calls disabled — with parallel=true the model may
      // call writeFile at the same time as createFolder, before the folder
      // exists, causing silent write failures.
      parallel_tool_calls: false,
      max_tokens: 4096,
    });

    const msg = response.choices[0].message;

    const assistantEntry = { role: "assistant", content: msg.content ?? null };
    if (msg.tool_calls?.length > 0) {
      assistantEntry.tool_calls = msg.tool_calls;
    }
    history.push(assistantEntry);

    if (msg.tool_calls?.length > 0) {
      for (const call of msg.tool_calls) {
        let args;
        try {
          args = JSON.parse(call.function.arguments);
        } catch {
          history.push({
            role: "tool",
            tool_call_id: call.id,
            content: "Error: Could not parse tool arguments as JSON.",
          });
          continue;
        }

        let result = "";
        if (call.function.name === "createFolder") {
          onEvent("command", `createFolder("${args.name}")`);
          result = await createFolder(args.name, sessionId);
          onEvent("result", result);
        } else if (call.function.name === "writeFile") {
          onEvent(
            "command",
            `writeFile("${args.filename}", <${args.content?.length ?? 0} chars>)`
          );
          result = await writeFile(args.filename, args.content ?? "", sessionId);
          onEvent("result", result);
        }

        history.push({ role: "tool", tool_call_id: call.id, content: result });
      }
    } else {
      if (msg.content) onEvent("response", msg.content);
      active = false;
    }
  }

  await launchProject(sessionId, onEvent);
}

// ─── POST /chat ───────────────────────────────────────────────────────────────
app.post("/chat", async (req, res) => {
  const { message } = req.body;
  if (!message) return res.status(400).json({ error: "message is required" });

  // FIX: Each request gets a brand-new session ID so:
  //  (a) a fresh folder is created for every prompt, and
  //  (b) projectDirs map never returns a stale old folder for a new build.
  const sid = `sid_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`;

  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");
  res.flushHeaders();

  const emit = (type, data) =>
    res.write(`data: ${JSON.stringify({ type, data })}\n\n`);

  // FIX: Do NOT pass old conversation history. Each website build is a
  // completely fresh task. Passing stale tool_call results from a previous
  // build causes the model to believe it already completed the work and
  // output nothing (which is exactly what was seen in Issue 2).
  const history = [
    { role: "system", content: SYSTEM_PROMPT },
    { role: "user", content: message },
  ];

  try {
    await runAgentTurn(history, sid, emit);
    emit("done", sid);
  } catch (err) {
    console.error("Agent error:", err);
    emit("error", err.message);
  } finally {
    res.end();
  }
});

// ─── UI ───────────────────────────────────────────────────────────────────────
app.get("/", (_req, res) => {
  res.send(`<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Craftly</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <!-- Classic, elegant typography -->
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Playfair+Display:ital,wght@0,600;0,700;1,600&family=Fira+Code:wght@400;500&display=swap" rel="stylesheet">
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      /* Bubbly Dark Theme */
      --bg: #0f111a; 
      --surface: rgba(22, 24, 33, 0.75); 
      --accent:  #a78bfa; 
      --accent2: #f472b6; 
      --accent-gradient: linear-gradient(135deg, #f472b6 0%, #a78bfa 100%);
      --text: #f8fafc;
      --muted: #94a3b8;
      --error: #fb7185;
      --success: #34d399;
      --radius: 32px; 
      --font-main: 'Inter', sans-serif;
      --font-head: 'Playfair Display', serif;
      --font-mono: 'Fira Code', monospace;
      --shadow-bubble: 0 12px 40px -12px rgba(244, 114, 182, 0.3), 0 0 20px rgba(167, 139, 250, 0.15);
    }

    html, body {
      height: 100%;
      background: var(--bg);
      color: var(--text);
      font-family: var(--font-main);
      -webkit-font-smoothing: antialiased;
    }

    body {
      display: flex;
      flex-direction: column;
      align-items: center;
      min-height: 100vh;
      padding: 40px 24px;
      /* Soft floating gradient blobs */
      background-image: 
        radial-gradient(circle at 10% 20%, rgba(244, 114, 182, 0.15) 0%, transparent 40%),
        radial-gradient(circle at 90% 80%, rgba(167, 139, 250, 0.15) 0%, transparent 40%),
        radial-gradient(circle at 50% 50%, rgba(56, 189, 248, 0.1) 0%, transparent 60%);
      background-attachment: fixed;
    }

    /* ── Header ── */
    header {
      width: 100%;
      max-width: 800px;
      margin-bottom: 24px;
      display: flex;
      align-items: center;
      gap: 18px;
      background: var(--surface);
      backdrop-filter: blur(12px);
      -webkit-backdrop-filter: blur(12px);
      padding: 16px 24px;
      border-radius: 100px;
      box-shadow: 0 4px 30px rgba(0,0,0,0.3);
      border: 2px solid rgba(255, 255, 255, 0.08); /* Dark subtle border */
    }
    .logo-badge {
      width: 54px; height: 54px;
      background: var(--accent-gradient);
      border-radius: 50%;
      display: flex; align-items: center; justify-content: center;
      font-size: 26px;
      flex-shrink: 0;
      box-shadow: 0 6px 16px rgba(244, 114, 182, 0.4);
      color: white;
    }
    .header-text { flex: 1; }
    .header-text h1 {
      font-family: var(--font-head);
      font-size: 26px;
      font-weight: 700;
      letter-spacing: -0.5px;
      background: var(--accent-gradient);
      -webkit-background-clip: text;
      background-clip: text;
      -webkit-text-fill-color: transparent;
    }
    .header-text p {
      font-size: 14px;
      font-weight: 700;
      color: var(--muted);
      margin-top: 2px;
    }

    /* ── Log pane ── */
    #log {
      width: 100%;
      max-width: 800px;
      background: var(--surface);
      backdrop-filter: blur(12px);
      -webkit-backdrop-filter: blur(12px);
      border: 2px solid rgba(255, 255, 255, 0.08);
      border-radius: var(--radius);
      padding: 28px 32px;
      height: 440px;
      overflow-y: auto;
      font-family: var(--font-mono);
      font-size: 14px;
      line-height: 1.8;
      margin-bottom: 24px;
      box-shadow: var(--shadow-bubble);
    }
    #log::-webkit-scrollbar { width: 8px; }
    #log::-webkit-scrollbar-track { background: transparent; }
    #log::-webkit-scrollbar-thumb { background: rgba(244, 114, 182, 0.4); border-radius: 10px; }
    #log::-webkit-scrollbar-thumb:hover { background: rgba(244, 114, 182, 0.7); }

    .log-line { display: flex; align-items: baseline; gap: 12px; padding: 4px 0; }
    .log-icon { 
      flex-shrink: 0; width: 24px; height: 24px; 
      display: inline-flex; align-items: center; justify-content: center; 
      border-radius: 50%; font-size: 12px; font-weight: bold; 
      transform: translateY(2px);
    }

    .log-you     .log-icon::before { content: "✨"; font-size: 14px; }
    .log-you     span:last-child    { color: var(--accent2); font-weight: 700; font-family: var(--font-main); font-size: 15px;}

    .log-status  .log-icon { background: rgba(255, 255, 255, 0.1); }
    .log-status  .log-icon::before { content: "·"; color: var(--muted); font-size: 16px; }
    .log-status  span:last-child    { color: var(--muted); font-family: var(--font-main); font-weight: 600; }

    .log-command .log-icon { background: rgba(59, 130, 246, 0.2); }
    .log-command .log-icon::before { content: "⚙"; color: #60a5fa; }
    .log-command span:last-child    { color: #60a5fa; font-weight: 500; }

    .log-result  .log-icon::before { content: "↳"; color: #475569; }
    .log-result  span:last-child    { color: #94a3b8; font-size: 13px; }

    .log-response .log-icon { background: rgba(16, 185, 129, 0.2); }
    .log-response .log-icon::before { content: "✓"; color: var(--success); }
    .log-response span:last-child    { color: var(--success); font-weight: 600; }

    .log-error   .log-icon { background: rgba(244, 63, 94, 0.2); }
    .log-error   .log-icon::before { content: "✗"; color: var(--error); }
    .log-error   span:last-child    { color: var(--error); font-weight: 600; }

    .log-launch {
      margin: 16px 0;
      padding: 16px 20px;
      background: rgba(245, 158, 11, 0.15);
      border: 2px solid rgba(245, 158, 11, 0.3);
      border-radius: 20px;
      color: #fbbf24;
      font-size: 14px;
      font-weight: 800;
      font-family: var(--font-main);
      display: inline-block;
      box-shadow: 0 4px 12px rgba(245, 158, 11, 0.1);
    }

    hr.log-divider {
      border: none;
      border-top: 2px dashed rgba(255, 255, 255, 0.1);
      margin: 20px 0;
    }

    .log-empty {
      height: 100%;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 16px;
      color: #94a3b8;
      pointer-events: none;
      user-select: none;
      animation: float 4s ease-in-out infinite;
    }
    @keyframes float {
      0%, 100% { transform: translateY(0px); }
      50% { transform: translateY(-12px); }
    }
    .log-empty-icon { font-size: 54px; filter: drop-shadow(0 8px 16px rgba(0,0,0,0.3)); }
    .log-empty-text { font-size: 16px; font-family: var(--font-main); font-weight: 700; color: #475569; }

    /* ── Input ── */
    .input-wrapper {
      width: 100%;
      max-width: 800px;
      background: var(--surface);
      backdrop-filter: blur(12px);
      -webkit-backdrop-filter: blur(12px);
      border: 2px solid rgba(255, 255, 255, 0.08);
      border-radius: var(--radius);
      overflow: hidden;
      transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
      box-shadow: var(--shadow-bubble);
    }
    .input-wrapper:focus-within {
      border-color: rgba(244, 114, 182, 0.5);
      box-shadow: 0 0 0 4px rgba(236, 72, 153, 0.15), var(--shadow-bubble);
      transform: translateY(-2px);
    }
    textarea {
      width: 100%;
      background: transparent;
      color: var(--text);
      border: none;
      outline: none;
      resize: none;
      padding: 24px 28px 12px;
      font-family: var(--font-main);
      font-size: 16px;
      font-weight: 700;
      line-height: 1.6;
      min-height: 110px;
    }
    textarea::placeholder { color: #64748b; font-weight: 600; }

    .input-footer {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 14px 20px 18px 28px;
      background: transparent;
    }
    .hint { font-size: 13px; color: #64748b; font-weight: 700; }
    .actions { display: flex; gap: 12px; }

    button {
      font-family: var(--font-main);
      font-weight: 800;
      font-size: 15px;
      border: none;
      border-radius: 100px; /* Pill shape */
      cursor: pointer;
      padding: 12px 28px;
      transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    }
    #sendBtn {
      background: var(--accent-gradient);
      color: #ffffff;
      box-shadow: 0 6px 16px rgba(236, 72, 153, 0.3);
    }
    #sendBtn:hover:not(:disabled) {
      transform: translateY(-2px) scale(1.03);
      box-shadow: 0 10px 24px rgba(236, 72, 153, 0.4);
    }
    #sendBtn:active:not(:disabled) {
      transform: translateY(1px);
    }
    #sendBtn:disabled { opacity: 0.6; cursor: not-allowed; transform: none; box-shadow: none; filter: grayscale(50%); }
    #sendBtn.loading { position: relative; color: transparent !important; pointer-events: none; }
    #sendBtn.loading::after {
      content: '';
      position: absolute;
      inset: 0; margin: auto;
      width: 20px; height: 20px;
      border: 4px solid rgba(255,255,255,.8);
      border-top-color: transparent;
      border-radius: 50%;
      animation: spin .8s cubic-bezier(0.4, 0, 0.2, 1) infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }

    #clearBtn {
      background: rgba(255, 255, 255, 0.1);
      color: #cbd5e1;
      padding: 12px 24px;
    }
    #clearBtn:hover { 
      background: rgba(255, 255, 255, 0.2);
      color: #ffffff; 
      transform: translateY(-2px);
    }

    /* ── Preview card ── */
    #previewCard {
      display: none;
      width: 100%;
      max-width: 800px;
      margin-top: 24px;
      background: linear-gradient(135deg, rgba(30, 20, 30, 0.8) 0%, rgba(35, 20, 25, 0.8) 100%);
      border: 2px solid rgba(244, 114, 182, 0.4);
      border-radius: var(--radius);
      padding: 24px 32px;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      box-shadow: var(--shadow-bubble);
      backdrop-filter: blur(12px);
      -webkit-backdrop-filter: blur(12px);
      animation: popIn 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275) forwards;
    }
    @keyframes popIn {
      from { opacity: 0; transform: scale(0.9) translateY(20px); }
      to { opacity: 1; transform: scale(1) translateY(0); }
    }
    #previewCard.visible { display: flex; }
    .preview-label { font-size: 16px; color: #fbcfe8; font-weight: 700; font-family: var(--font-main); }
    .preview-label strong { color: #f472b6; font-weight: 800; background: rgba(0,0,0,0.3); padding: 4px 10px; border-radius: 8px; margin-left: 6px;}
    #previewLink {
      background: var(--accent-gradient);
      color: #ffffff;
      font-family: var(--font-main);
      font-weight: 800;
      font-size: 15px;
      padding: 12px 28px;
      border-radius: 100px;
      text-decoration: none;
      flex-shrink: 0;
      transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
      box-shadow: 0 6px 16px rgba(236, 72, 153, 0.3);
    }
    #previewLink:hover { transform: translateY(-2px) scale(1.05); box-shadow: 0 10px 24px rgba(236, 72, 153, 0.4); }
  </style>
</head>
<body>

<header>
  <div class="logo-badge">🎨</div>
  <div class="header-text">
    <h1>Craftly</h1>
    <p>AI-powered website builder</p>
  </div>
</header>

<div id="log">
  <div class="log-empty">
    <div class="log-empty-icon">✨</div>
    <div class="log-empty-text">Describe a website to get started</div>
  </div>
</div>

<div class="input-wrapper">
  <textarea id="input" rows="3" placeholder="e.g. a minimal portfolio for a motion designer..."></textarea>
  <div class="input-footer">
    <span class="hint">Ctrl+Enter to send</span>
    <div class="actions">
      <button id="clearBtn">Clear</button>
      <button id="sendBtn">Send →</button>
    </div>
  </div>
</div>

<div id="previewCard">
  <div class="preview-label">Website ready → <strong id="previewUrl"></strong></div>
  <a id="previewLink" href="#" target="_blank">Open Site ↗</a>
</div>

<script>
  const logEl       = document.getElementById('log');
  const inputEl     = document.getElementById('input');
  const sendBtn     = document.getElementById('sendBtn');
  const clearBtn    = document.getElementById('clearBtn');
  const previewCard = document.getElementById('previewCard');
  const previewLink = document.getElementById('previewLink');
  const previewUrl  = document.getElementById('previewUrl');

  function clearEmpty() {
    const e = logEl.querySelector('.log-empty');
    if (e) e.remove();
  }

  function appendLog(type, text) {
    clearEmpty();

    if (type === 'divider') {
      const hr = document.createElement('hr');
      hr.className = 'log-divider';
      logEl.appendChild(hr);
    } else if (type === 'launch') {
      const div = document.createElement('div');
      div.className = 'log-launch';
      div.textContent = '🚀 ' + text;
      logEl.appendChild(div);
    } else {
      const line = document.createElement('div');
      line.className = 'log-line log-' + type;
      const icon = document.createElement('span');
      icon.className = 'log-icon';
      const msg = document.createElement('span');
      msg.textContent = text;
      line.appendChild(icon);
      line.appendChild(msg);
      logEl.appendChild(line);
    }
    logEl.scrollTop = logEl.scrollHeight;
  }

  clearBtn.addEventListener('click', () => {
    logEl.innerHTML = '<div class="log-empty"><div class="log-empty-icon">✨</div><div class="log-empty-text">Describe a website to get started</div></div>';
    previewCard.classList.remove('visible');
    inputEl.value = '';
  });

  inputEl.addEventListener('keydown', e => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) sendBtn.click();
  });

  sendBtn.addEventListener('click', async () => {
    const msg = inputEl.value.trim();
    if (!msg || sendBtn.disabled) return;

    appendLog('divider', '');
    appendLog('you', msg);
    inputEl.value = '';
    sendBtn.disabled = true;
    sendBtn.classList.add('loading');
    previewCard.classList.remove('visible');

    try {
      const res = await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: msg }),
      });

      const reader  = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split('\\n\\n');
        buffer = parts.pop();
        for (const part of parts) {
          if (!part.startsWith('data:')) continue;
          let parsed;
          try { parsed = JSON.parse(part.slice(5).trim()); } catch { continue; }
          const { type, data } = parsed;

          if      (type === 'status')   appendLog('status',   data);
          else if (type === 'command')  appendLog('command',  data);
          else if (type === 'result')   appendLog('result',   data);
          else if (type === 'response') appendLog('response', data);
          else if (type === 'error')    appendLog('error',    data);
          else if (type === 'launch') {
            appendLog('launch', 'Website ready at ' + data);
            previewLink.href = data;
            previewUrl.textContent = data;
            previewCard.classList.add('visible');
          }
        }
      }
    } catch (err) {
      appendLog('error', 'Network error: ' + err.message);
    } finally {
      sendBtn.disabled = false;
      sendBtn.classList.remove('loading');
    }
  });
</script>
</body>
</html>
`);
});

// ─── Start ────────────────────────────────────────────────────────────────────
const PORT = process.env.PORT || 3000;
httpServer.listen(PORT, () => {
  console.log(`\n🚀 Groq Website Builder at: http://localhost:${PORT}`);
});