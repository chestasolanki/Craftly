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
      model: "meta-llama/llama-4-scout-17b-16e-instruct",
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
  <link href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=Fraunces:opsz,wght,SOFT@9..144,600,100&family=JetBrains+Mono:wght@500&display=swap" rel="stylesheet">
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --bg: #0a0d12;
      --bg-soft: #10151d;
      --surface: rgba(15, 20, 28, 0.74);
      --surface-strong: rgba(18, 24, 33, 0.94);
      --surface-elevated: rgba(22, 29, 40, 0.92);
      --border: rgba(255, 255, 255, 0.08);
      --border-strong: rgba(255, 255, 255, 0.14);
      --text: #f5f7fb;
      --muted: #95a2b7;
      --muted-soft: #6d7a90;
      --accent: #7dd3fc;
      --accent-strong: #37b7ff;
      --accent-warm: #d6c3a1;
      --success: #4ade80;
      --error: #fb7185;
      --radius-xl: 32px;
      --radius-lg: 24px;
      --radius-md: 18px;
      --radius-sm: 14px;
      --shadow-lg: 0 30px 80px rgba(0, 0, 0, 0.45);
      --shadow-md: 0 18px 42px rgba(0, 0, 0, 0.28);
      --glow: 0 0 0 1px rgba(255,255,255,0.04), 0 20px 40px rgba(9, 14, 22, 0.45);
      --font-main: 'Manrope', sans-serif;
      --font-head: 'Fraunces', serif;
      --font-mono: 'JetBrains Mono', monospace;
      --accent-gradient: linear-gradient(135deg, #d8c29d 0%, #7dd3fc 48%, #3b82f6 100%);
      --panel-gradient: linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.015));
    }

    html, body {
      min-height: 100%;
      background: var(--bg);
      color: var(--text);
      font-family: var(--font-main);
      -webkit-font-smoothing: antialiased;
      text-rendering: optimizeLegibility;
    }

    body {
      position: relative;
      overflow-x: hidden;
      min-height: 100vh;
      padding: 32px 20px 40px;
      background:
        radial-gradient(circle at top left, rgba(125, 211, 252, 0.16), transparent 30%),
        radial-gradient(circle at 85% 15%, rgba(214, 195, 161, 0.12), transparent 24%),
        radial-gradient(circle at 50% 100%, rgba(56, 189, 248, 0.1), transparent 28%),
        linear-gradient(180deg, #091019 0%, #0a0d12 50%, #0c1017 100%);
    }

    body::before,
    body::after {
      content: "";
      position: fixed;
      inset: auto;
      pointer-events: none;
      z-index: 0;
    }

    body::before {
      top: -140px;
      right: -80px;
      width: 380px;
      height: 380px;
      border-radius: 50%;
      background: radial-gradient(circle, rgba(125, 211, 252, 0.12), transparent 70%);
      filter: blur(8px);
      animation: drift 14s ease-in-out infinite;
    }

    body::after {
      bottom: -140px;
      left: -110px;
      width: 420px;
      height: 420px;
      border-radius: 50%;
      background: radial-gradient(circle, rgba(214, 195, 161, 0.11), transparent 72%);
      filter: blur(10px);
      animation: drift 18s ease-in-out infinite reverse;
    }

    .shell {
      position: relative;
      z-index: 1;
      width: min(1120px, 100%);
      margin: 0 auto;
      display: grid;
      gap: 22px;
    }

    .hero {
      position: relative;
      overflow: hidden;
      display: grid;
      grid-template-columns: minmax(0, 1.3fr) minmax(280px, 0.8fr);
      gap: 18px;
      padding: 18px;
      border-radius: 36px;
      border: 1px solid var(--border);
      background:
        linear-gradient(135deg, rgba(18, 24, 33, 0.9), rgba(9, 13, 18, 0.92)),
        var(--panel-gradient);
      box-shadow: var(--shadow-lg);
      backdrop-filter: blur(20px);
      -webkit-backdrop-filter: blur(20px);
    }

    .hero::before {
      content: "";
      position: absolute;
      inset: 0;
      background:
        linear-gradient(115deg, rgba(255,255,255,0.06), transparent 35%),
        radial-gradient(circle at top right, rgba(125, 211, 252, 0.14), transparent 24%);
      pointer-events: none;
    }

    .hero-main,
    .hero-side {
      position: relative;
      border-radius: 28px;
      border: 1px solid rgba(255,255,255,0.06);
      background: rgba(255,255,255,0.025);
    }

    .hero-main {
      padding: 28px;
    }

    .hero-topbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 24px;
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 14px;
    }

    .logo-badge {
      width: 58px;
      height: 58px;
      border-radius: 18px;
      display: grid;
      place-items: center;
      font-size: 26px;
      color: #071018;
      background: var(--accent-gradient);
      box-shadow: 0 20px 38px rgba(55, 183, 255, 0.2);
    }

    .brand-copy h1 {
      font-family: var(--font-head);
      font-size: clamp(1.8rem, 2.8vw, 2.45rem);
      font-weight: 600;
      line-height: 1;
      letter-spacing: -0.03em;
    }

    .brand-copy p {
      margin-top: 6px;
      color: var(--muted);
      font-size: 0.95rem;
      font-weight: 600;
      letter-spacing: 0.01em;
    }

    .hero-chip {
      display: inline-flex;
      align-items: center;
      gap: 10px;
      padding: 10px 14px;
      border-radius: 999px;
      border: 1px solid rgba(125, 211, 252, 0.22);
      background: rgba(125, 211, 252, 0.08);
      color: #dff4ff;
      font-size: 0.78rem;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      backdrop-filter: blur(16px);
      -webkit-backdrop-filter: blur(16px);
    }

    .hero-chip::before {
      content: "";
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: #8ee7ff;
      box-shadow: 0 0 18px rgba(142, 231, 255, 0.8);
    }

    .hero-copy {
      max-width: 720px;
    }

    .hero-copy h2 {
      font-size: clamp(2.25rem, 5vw, 4.6rem);
      line-height: 0.96;
      letter-spacing: -0.06em;
      font-weight: 800;
      margin-bottom: 16px;
    }

    .hero-copy .accent {
      display: block;
      font-family: var(--font-head);
      font-weight: 600;
      background: var(--accent-gradient);
      -webkit-background-clip: text;
      background-clip: text;
      -webkit-text-fill-color: transparent;
    }

    .hero-copy p {
      max-width: 650px;
      font-size: 1.02rem;
      line-height: 1.75;
      color: var(--muted);
    }

    .hero-meta {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin-top: 24px;
    }

    .meta-pill {
      padding: 12px 16px;
      border-radius: 999px;
      background: rgba(255,255,255,0.05);
      border: 1px solid rgba(255,255,255,0.08);
      color: #d7dfec;
      font-size: 0.84rem;
      font-weight: 600;
      transition: transform 220ms ease, border-color 220ms ease, background 220ms ease;
    }

    .meta-pill:hover {
      transform: translateY(-2px);
      border-color: rgba(125, 211, 252, 0.22);
      background: rgba(125, 211, 252, 0.08);
    }

    .hero-side {
      padding: 24px;
      display: grid;
      grid-template-rows: auto 1fr auto;
      gap: 18px;
      min-height: 100%;
    }

    .side-label {
      color: var(--muted-soft);
      font-size: 0.78rem;
      font-weight: 800;
      letter-spacing: 0.12em;
      text-transform: uppercase;
    }

    .signal-card {
      padding: 18px;
      border-radius: 22px;
      background: rgba(255,255,255,0.04);
      border: 1px solid rgba(255,255,255,0.06);
      box-shadow: var(--glow);
    }

    .signal-card strong {
      display: block;
      font-size: 1.15rem;
      margin-top: 8px;
    }

    .signal-card p {
      margin-top: 8px;
      color: var(--muted);
      line-height: 1.6;
      font-size: 0.94rem;
    }

    .signal-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }

    .mini-stat {
      padding: 16px;
      border-radius: 18px;
      background: rgba(255,255,255,0.035);
      border: 1px solid rgba(255,255,255,0.06);
      transition: transform 220ms ease, border-color 220ms ease, background 220ms ease;
    }

    .mini-stat:hover {
      transform: translateY(-3px);
      border-color: rgba(214, 195, 161, 0.2);
      background: rgba(214, 195, 161, 0.07);
    }

    .mini-stat span {
      display: block;
      color: var(--muted-soft);
      font-size: 0.72rem;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }

    .mini-stat strong {
      display: block;
      margin-top: 8px;
      font-size: 0.95rem;
      line-height: 1.45;
    }

    .workspace {
      display: grid;
      grid-template-columns: minmax(0, 1.2fr) minmax(320px, 0.82fr);
      gap: 22px;
      align-items: start;
    }

    .panel {
      position: relative;
      overflow: hidden;
      border-radius: var(--radius-xl);
      border: 1px solid var(--border);
      background:
        linear-gradient(180deg, rgba(18, 24, 33, 0.92), rgba(10, 14, 20, 0.96)),
        var(--panel-gradient);
      box-shadow: var(--shadow-md);
      backdrop-filter: blur(18px);
      -webkit-backdrop-filter: blur(18px);
    }

    .panel::before {
      content: "";
      position: absolute;
      inset: 0;
      background: linear-gradient(180deg, rgba(255,255,255,0.045), transparent 20%);
      pointer-events: none;
    }

    .panel-head {
      position: relative;
      z-index: 1;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 24px 26px 18px;
      border-bottom: 1px solid rgba(255,255,255,0.06);
    }

    .panel-title h3 {
      font-size: 1.08rem;
      letter-spacing: -0.03em;
      font-weight: 800;
    }

    .panel-title p {
      margin-top: 5px;
      color: var(--muted);
      font-size: 0.9rem;
    }

    .status-dot {
      display: inline-flex;
      align-items: center;
      gap: 10px;
      color: #dcefff;
      font-size: 0.78rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.12em;
    }

    .status-dot::before {
      content: "";
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: #7dd3fc;
      box-shadow: 0 0 18px rgba(125, 211, 252, 0.8);
      animation: pulse 1.8s infinite ease-in-out;
    }

    #log {
      position: relative;
      z-index: 1;
      height: 520px;
      overflow-y: auto;
      padding: 20px 26px 28px;
      font-family: var(--font-mono);
      font-size: 13px;
      line-height: 1.7;
      scroll-behavior: smooth;
    }

    #log::-webkit-scrollbar { width: 10px; }
    #log::-webkit-scrollbar-track { background: transparent; }
    #log::-webkit-scrollbar-thumb {
      background: linear-gradient(180deg, rgba(125, 211, 252, 0.28), rgba(214, 195, 161, 0.18));
      border-radius: 999px;
      border: 2px solid transparent;
      background-clip: padding-box;
    }

    .log-line {
      display: flex;
      align-items: flex-start;
      gap: 14px;
      margin-bottom: 12px;
      padding: 14px 16px;
      border-radius: 18px;
      border: 1px solid rgba(255,255,255,0.04);
      background: rgba(255,255,255,0.025);
      animation: messageIn 360ms cubic-bezier(.2,.8,.2,1);
      transition: transform 220ms ease, border-color 220ms ease, background 220ms ease;
    }

    .log-line:hover {
      transform: translateY(-2px);
      border-color: rgba(255,255,255,0.08);
      background: rgba(255,255,255,0.04);
    }

    .log-icon {
      width: 32px;
      height: 32px;
      flex-shrink: 0;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      border-radius: 12px;
      border: 1px solid rgba(255,255,255,0.07);
      background: rgba(255,255,255,0.04);
      transform: translateY(1px);
    }

    .log-line span:last-child {
      flex: 1;
      min-width: 0;
      word-break: break-word;
    }

    .log-you {
      background: linear-gradient(180deg, rgba(125, 211, 252, 0.09), rgba(125, 211, 252, 0.04));
      border-color: rgba(125, 211, 252, 0.12);
    }

    .log-you .log-icon::before { content: "✦"; color: #8ee7ff; font-size: 13px; }
    .log-you span:last-child {
      color: #edf8ff;
      font-family: var(--font-main);
      font-size: 0.98rem;
      font-weight: 700;
      line-height: 1.65;
    }

    .log-status .log-icon::before { content: "•"; color: var(--muted); font-size: 17px; }
    .log-status span:last-child {
      color: var(--muted);
      font-family: var(--font-main);
      font-weight: 600;
    }

    .log-command {
      background: linear-gradient(180deg, rgba(59, 130, 246, 0.09), rgba(59, 130, 246, 0.04));
      border-color: rgba(59, 130, 246, 0.12);
    }

    .log-command .log-icon::before { content: "⌘"; color: #93c5fd; font-size: 13px; }
    .log-command span:last-child { color: #bfdbfe; }

    .log-result .log-icon::before { content: "↳"; color: #9aa8ba; font-size: 14px; }
    .log-result span:last-child { color: #9eabc0; }

    .log-response {
      background: linear-gradient(180deg, rgba(74, 222, 128, 0.085), rgba(74, 222, 128, 0.035));
      border-color: rgba(74, 222, 128, 0.12);
    }

    .log-response .log-icon::before { content: "✓"; color: var(--success); font-size: 13px; }
    .log-response span:last-child {
      color: #d8ffe4;
      font-family: var(--font-main);
      font-weight: 700;
    }

    .log-error {
      background: linear-gradient(180deg, rgba(251, 113, 133, 0.09), rgba(251, 113, 133, 0.035));
      border-color: rgba(251, 113, 133, 0.13);
    }

    .log-error .log-icon::before { content: "!"; color: var(--error); font-size: 13px; }
    .log-error span:last-child {
      color: #ffd5de;
      font-family: var(--font-main);
      font-weight: 700;
    }

    .log-launch {
      display: flex;
      align-items: center;
      gap: 12px;
      margin: 6px 0 14px;
      padding: 16px 18px;
      border-radius: 20px;
      border: 1px solid rgba(214, 195, 161, 0.2);
      background: linear-gradient(180deg, rgba(214, 195, 161, 0.12), rgba(214, 195, 161, 0.05));
      color: #f1e6d3;
      font-family: var(--font-main);
      font-size: 0.93rem;
      font-weight: 700;
      animation: messageIn 360ms cubic-bezier(.2,.8,.2,1);
    }

    .log-launch::before {
      content: "↗";
      display: inline-grid;
      place-items: center;
      width: 30px;
      height: 30px;
      border-radius: 12px;
      background: rgba(214, 195, 161, 0.14);
      color: #f3ddbb;
      flex-shrink: 0;
    }

    hr.log-divider {
      border: none;
      border-top: 1px solid rgba(255,255,255,0.08);
      margin: 6px 0 18px;
    }

    .log-empty {
      min-height: 100%;
      display: grid;
      place-items: center;
      text-align: center;
      padding: 20px;
    }

    .empty-state {
      max-width: 420px;
      padding: 30px 24px;
      border-radius: 28px;
      border: 1px solid rgba(255,255,255,0.05);
      background: rgba(255,255,255,0.025);
      box-shadow: var(--glow);
      animation: float 6s ease-in-out infinite;
    }

    .log-empty-icon {
      font-size: 2.8rem;
      margin-bottom: 14px;
      filter: drop-shadow(0 12px 18px rgba(0,0,0,0.25));
    }

    .log-empty-text {
      font-size: 1.12rem;
      font-weight: 800;
      letter-spacing: -0.02em;
    }

    .log-empty-subtext {
      margin-top: 10px;
      color: var(--muted);
      line-height: 1.7;
      font-size: 0.95rem;
    }

    .composer {
      padding: 10px;
    }

    .input-wrapper {
      display: grid;
      gap: 0;
      border-radius: 30px;
      overflow: hidden;
      border: 1px solid var(--border);
      background:
        linear-gradient(180deg, rgba(19, 26, 36, 0.96), rgba(11, 15, 21, 0.96)),
        var(--panel-gradient);
      box-shadow: var(--shadow-md);
      transition: transform 260ms ease, border-color 260ms ease, box-shadow 260ms ease;
    }

    .input-wrapper:focus-within {
      transform: translateY(-2px);
      border-color: rgba(125, 211, 252, 0.2);
      box-shadow: 0 0 0 6px rgba(125, 211, 252, 0.08), var(--shadow-md);
    }

    .composer-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 18px 22px 0;
      color: var(--muted-soft);
      font-size: 0.76rem;
      font-weight: 800;
      letter-spacing: 0.12em;
      text-transform: uppercase;
    }

    .composer-head span:last-child {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      color: #d9eefe;
    }

    .composer-head span:last-child::before {
      content: "";
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background: #7dd3fc;
      box-shadow: 0 0 12px rgba(125, 211, 252, 0.9);
    }

    textarea {
      width: 100%;
      min-height: 150px;
      resize: none;
      border: none;
      outline: none;
      background: transparent;
      color: var(--text);
      padding: 18px 22px 12px;
      font-family: var(--font-main);
      font-size: 1rem;
      font-weight: 600;
      line-height: 1.8;
    }

    textarea::placeholder {
      color: #667389;
      font-weight: 600;
    }

    .input-footer {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 16px 22px 22px;
    }

    .hint {
      color: var(--muted);
      font-size: 0.9rem;
      line-height: 1.6;
    }

    .actions {
      display: flex;
      align-items: center;
      gap: 12px;
    }

    button,
    #previewLink {
      position: relative;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 10px;
      min-height: 50px;
      padding: 0 20px;
      border-radius: 999px;
      font-family: var(--font-main);
      font-size: 0.95rem;
      font-weight: 800;
      letter-spacing: -0.01em;
      text-decoration: none;
      cursor: pointer;
      transition: transform 220ms ease, box-shadow 220ms ease, border-color 220ms ease, background 220ms ease, color 220ms ease;
    }

    button:hover,
    #previewLink:hover {
      transform: translateY(-2px);
    }

    button:active,
    #previewLink:active {
      transform: translateY(0);
    }

    #sendBtn {
      min-width: 138px;
      border: 1px solid rgba(125, 211, 252, 0.24);
      color: #08111a;
      background: var(--accent-gradient);
      box-shadow: 0 14px 26px rgba(55, 183, 255, 0.24);
    }

    #sendBtn::after {
      content: "→";
      font-size: 1rem;
    }

    #sendBtn:hover:not(:disabled) {
      box-shadow: 0 20px 36px rgba(55, 183, 255, 0.28);
    }

    #sendBtn:disabled {
      opacity: 0.55;
      cursor: not-allowed;
      transform: none;
      box-shadow: none;
    }

    #sendBtn.loading {
      color: transparent !important;
      pointer-events: none;
    }

    #sendBtn.loading::after {
      content: "";
      position: absolute;
      width: 18px;
      height: 18px;
      border-radius: 50%;
      border: 3px solid rgba(8, 17, 26, 0.24);
      border-top-color: #08111a;
      animation: spin 0.85s linear infinite;
    }

    #clearBtn {
      border: 1px solid rgba(255,255,255,0.08);
      background: rgba(255,255,255,0.05);
      color: #dce5f2;
    }

    #clearBtn:hover {
      border-color: rgba(255,255,255,0.14);
      background: rgba(255,255,255,0.08);
    }

    #previewCard {
      display: none;
      align-items: center;
      justify-content: space-between;
      gap: 20px;
      padding: 22px 24px;
      border-radius: 28px;
      border: 1px solid rgba(125, 211, 252, 0.16);
      background:
        linear-gradient(135deg, rgba(14, 24, 36, 0.94), rgba(9, 14, 20, 0.96)),
        var(--panel-gradient);
      box-shadow: var(--shadow-md);
      animation: cardIn 420ms cubic-bezier(.2,.8,.2,1);
    }

    #previewCard.visible {
      display: flex;
    }

    .preview-copy {
      min-width: 0;
    }

    .preview-kicker {
      color: var(--muted-soft);
      font-size: 0.76rem;
      font-weight: 800;
      letter-spacing: 0.12em;
      text-transform: uppercase;
    }

    .preview-label {
      margin-top: 8px;
      font-size: 1.05rem;
      font-weight: 800;
      letter-spacing: -0.02em;
    }

    .preview-url {
      margin-top: 8px;
      color: #cae8ff;
      font-family: var(--font-mono);
      font-size: 0.84rem;
      word-break: break-all;
    }

    #previewLink {
      flex-shrink: 0;
      border: 1px solid rgba(214, 195, 161, 0.2);
      color: #f7efe0;
      background: rgba(214, 195, 161, 0.09);
    }

    #previewLink:hover {
      border-color: rgba(214, 195, 161, 0.34);
      background: rgba(214, 195, 161, 0.14);
    }

    @keyframes spin { to { transform: rotate(360deg); } }
    @keyframes pulse {
      0%, 100% { transform: scale(1); opacity: 0.9; }
      50% { transform: scale(1.25); opacity: 1; }
    }
    @keyframes float {
      0%, 100% { transform: translateY(0); }
      50% { transform: translateY(-10px); }
    }
    @keyframes drift {
      0%, 100% { transform: translate3d(0, 0, 0); }
      50% { transform: translate3d(18px, 12px, 0); }
    }
    @keyframes messageIn {
      from { opacity: 0; transform: translateY(12px) scale(0.985); }
      to { opacity: 1; transform: translateY(0) scale(1); }
    }
    @keyframes cardIn {
      from { opacity: 0; transform: translateY(14px); }
      to { opacity: 1; transform: translateY(0); }
    }

    @media (max-width: 980px) {
      .hero,
      .workspace {
        grid-template-columns: 1fr;
      }

      .hero-copy p {
        max-width: none;
      }
    }

    @media (max-width: 720px) {
      body {
        padding: 18px 14px 28px;
      }

      .hero {
        padding: 12px;
        border-radius: 28px;
      }

      .hero-main,
      .hero-side,
      .panel,
      .input-wrapper,
      #previewCard {
        border-radius: 22px;
      }

      .hero-main,
      .hero-side {
        padding: 20px;
      }

      .hero-topbar,
      .input-footer,
      .panel-head,
      #previewCard {
        flex-direction: column;
        align-items: flex-start;
      }

      .hero-copy h2 {
        font-size: clamp(2rem, 13vw, 3rem);
      }

      .signal-grid {
        grid-template-columns: 1fr;
      }

      #log {
        height: 420px;
        padding: 18px;
      }

      textarea {
        min-height: 136px;
        font-size: 0.98rem;
      }

      .actions {
        width: 100%;
      }

      .actions button {
        flex: 1;
      }

      #previewLink {
        width: 100%;
      }
    }
  </style>
</head>
<body>
<main class="shell">
  <section class="hero">
    <div class="hero-main">
      <div class="hero-topbar">
        <div class="brand">
          <div class="logo-badge">✦</div>
          <div class="brand-copy">
            <h1>Craftly</h1>
            <p>AI-powered website builder</p>
          </div>
        </div>
        <div class="hero-chip">Live Build Workspace</div>
      </div>

      <div class="hero-copy">
        <h2>
          Shape polished websites
          <span class="accent">from a single prompt</span>
        </h2>
        <p>
          A refined workspace for turning rough ideas into elegant frontends. Describe the site you want, then watch the build stream unfold in real time.
        </p>
      </div>

      <div class="hero-meta">
        <div class="meta-pill">Clean execution logs</div>
        <div class="meta-pill">Fast previews</div>
        <div class="meta-pill">Minimal, high-clarity UI</div>
      </div>
    </div>

    <aside class="hero-side">
      <span class="side-label">Studio Snapshot</span>
      <div class="signal-card">
        <span class="side-label">Current Mode</span>
        <strong>Prompt to production preview</strong>
        <p>Designed to feel calm, focused, and premium while the agent handles the heavy lifting behind the scenes.</p>
      </div>
      <div class="signal-grid">
        <div class="mini-stat">
          <span>Interface</span>
          <strong>Minimal by default</strong>
        </div>
        <div class="mini-stat">
          <span>Motion</span>
          <strong>Smooth, subtle feedback</strong>
        </div>
        <div class="mini-stat">
          <span>Layout</span>
          <strong>Responsive and airy</strong>
        </div>
        <div class="mini-stat">
          <span>Tone</span>
          <strong>Modern AI product feel</strong>
        </div>
      </div>
    </aside>
  </section>

  <section class="workspace">
    <div class="panel">
      <div class="panel-head">
        <div class="panel-title">
          <h3>Build Activity</h3>
          <p>Real-time execution, file writes, status updates, and launch events.</p>
        </div>
        <div class="status-dot">Streaming</div>
      </div>

      <div id="log">
        <div class="log-empty">
          <div class="empty-state">
            <div class="log-empty-icon">✦</div>
            <div class="log-empty-text">Describe a website to get started</div>
            <div class="log-empty-subtext">Your prompt appears here first, followed by each step the builder takes to assemble and launch the preview.</div>
          </div>
        </div>
      </div>
    </div>

    <div class="composer">
      <div class="input-wrapper">
        <div class="composer-head">
          <span>Project Brief</span>
          <span>Ready</span>
        </div>
        <textarea id="input" rows="3" placeholder="Describe the website you want. Try: a cinematic landing page for an AI design startup with soft gradients and elegant typography..."></textarea>
        <div class="input-footer">
          <span class="hint">Press Ctrl+Enter to send and generate a fresh preview.</span>
          <div class="actions">
            <button id="clearBtn">Clear</button>
            <button id="sendBtn">Generate</button>
          </div>
        </div>
      </div>

      <div id="previewCard">
        <div class="preview-copy">
          <div class="preview-kicker">Preview Ready</div>
          <div class="preview-label">Your website has been launched.</div>
          <div class="preview-url" id="previewUrl"></div>
        </div>
        <a id="previewLink" href="#" target="_blank">Open Preview</a>
      </div>
    </div>
  </section>
</main>

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

  function emptyMarkup() {
    return '<div class="log-empty"><div class="empty-state"><div class="log-empty-icon">✦</div><div class="log-empty-text">Describe a website to get started</div><div class="log-empty-subtext">Your prompt appears here first, followed by each step the builder takes to assemble and launch the preview.</div></div></div>';
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
    logEl.innerHTML = emptyMarkup();
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
const PORT = process.env.PORT || 8080;

app.listen(PORT, "0.0.0.0", () => {
  console.log(`🚀 Server running on port ${PORT}`);
});
