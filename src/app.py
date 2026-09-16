#!/usr/bin/env python3
"""
Support Triage Agent — Flask Web App
=====================================
Classifies incoming support tickets (category + sentiment), retrieves
relevant company policy passages via a local ChromaDB vector store (RAG),
and drafts a suggested reply grounded in those passages — via Groq's free
hosted LLM API.

Local run:   python app.py
Open:        http://localhost:5000
"""
import os
import time
import uuid
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from flask import Flask, jsonify, request, session

load_dotenv()

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import triage_core

BASE_DIR = Path(__file__).resolve().parent.parent

MAX_UPLOAD_BYTES = 2 * 1024 * 1024   # 2 MB
MAX_ROWS = 50                         # cap per batch for a public demo

MAX_SESSIONS = 20
SESSION_TTL_SECONDS = 30 * 60
SESSION_STORE = {}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
app.secret_key = os.environ.get("SECRET_KEY", "dev-only-insecure-key-change-in-prod")


@app.errorhandler(413)
def too_large(_e):
    return jsonify({"error": f"File is larger than the {MAX_UPLOAD_BYTES // (1024*1024)}MB limit."}), 413


def _prune_sessions():
    now = time.time()
    expired = [sid for sid, entry in SESSION_STORE.items()
               if now - entry["created"] > SESSION_TTL_SECONDS]
    for sid in expired:
        SESSION_STORE.pop(sid, None)
    if len(SESSION_STORE) > MAX_SESSIONS:
        oldest = sorted(SESSION_STORE.items(), key=lambda kv: kv[1]["created"])
        for sid, _ in oldest[: len(SESSION_STORE) - MAX_SESSIONS]:
            SESSION_STORE.pop(sid, None)


def get_session_id():
    sid = session.get("sid")
    if not sid:
        sid = str(uuid.uuid4())
        session["sid"] = sid
    return sid


@app.route("/")
def index():
    return HTML_TEMPLATE


@app.route("/api/sample")
def api_sample():
    sample_path = BASE_DIR / "sample_data" / "sample_tickets.csv"
    return sample_path.read_text(encoding="utf-8"), 200, {"Content-Type": "text/csv"}


@app.route("/api/triage-one", methods=["POST"])
def api_triage_one():
    body = request.get_json(silent=True) or {}
    subject = (body.get("subject") or "").strip()
    text = (body.get("body") or "").strip()
    if not text:
        return jsonify({"error": "Ticket body is required."}), 400
    try:
        result = triage_core.triage_one(subject, text)
        return jsonify({"success": True, "result": result})
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": f"Something went wrong: {e}"}), 500


@app.route("/api/triage-batch", methods=["POST"])
def api_triage_batch():
    _prune_sessions()
    sid = get_session_id()

    use_sample = request.form.get("use_sample") == "1"
    try:
        if use_sample:
            df = pd.read_csv(BASE_DIR / "sample_data" / "sample_tickets.csv")
        else:
            if "file" not in request.files:
                return jsonify({"error": "No file provided."}), 400
            f = request.files["file"]
            df = pd.read_csv(f)

        required_cols = {"subject", "body"}
        cols_lower = {c.lower(): c for c in df.columns}
        if not required_cols.issubset(cols_lower.keys()):
            return jsonify({"error": "CSV must have 'subject' and 'body' columns."}), 400

        if len(df) > MAX_ROWS:
            df = df.head(MAX_ROWS)
            truncated = True
        else:
            truncated = False

        results = []
        for _, row in df.iterrows():
            subject = str(row[cols_lower["subject"]])
            text = str(row[cols_lower["body"]])
            r = triage_core.triage_one(subject, text)
            results.append({"subject": subject, "body": text, **r})

        SESSION_STORE[sid] = {"created": time.time()}

        return jsonify({"success": True, "results": results, "truncated": truncated})

    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": f"Something went wrong processing this file: {e}"}), 500


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Support Triage Agent</title>
<meta name="description" content="Classifies support tickets, retrieves relevant policy, and drafts a grounded reply.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700&family=Spline+Sans+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root {
  --ground: #FAF9F6; --ground-raised: #F2F0E9; --ink: #1E1B16; --ink-soft: #6B6355;
  --line: #DDD6C8; --line-soft: #E9E4D8; --accent: #3D5A45; --accent-soft: #E4EBE3;
  --amber: #B5762A; --amber-soft: #F5EBDC; --red: #A44430; --red-soft: #F5E5E0;
  --chip-bg: #FFFFFF; --shadow: 0 1px 2px rgba(30,27,22,.06), 0 4px 16px rgba(30,27,22,.04);
  --serif: 'Source Serif 4', Georgia, serif; --mono: 'Spline Sans Mono', ui-monospace, 'SF Mono', Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --ground:#17160F; --ground-raised:#1E1C14; --ink:#EDE9DE; --ink-soft:#A39B87;
    --line:#383426; --line-soft:#2A271C; --accent:#7FA089; --accent-soft:#22291F;
    --amber:#D99A4E; --amber-soft:#2C2416; --red:#D97A63; --red-soft:#2C1E19;
    --chip-bg:#1E1C14; --shadow: 0 1px 2px rgba(0,0,0,.3), 0 4px 20px rgba(0,0,0,.25);
  }
}
:root[data-theme="dark"] {
  --ground:#17160F; --ground-raised:#1E1C14; --ink:#EDE9DE; --ink-soft:#A39B87;
  --line:#383426; --line-soft:#2A271C; --accent:#7FA089; --accent-soft:#22291F;
  --amber:#D99A4E; --amber-soft:#2C2416; --red:#D97A63; --red-soft:#2C1E19;
  --chip-bg:#1E1C14; --shadow: 0 1px 2px rgba(0,0,0,.3), 0 4px 20px rgba(0,0,0,.25);
}
* { box-sizing: border-box; }
body { background: var(--ground); color: var(--ink); font-family: var(--serif); margin: 0; padding: 0 20px; padding-block: 40px 80px; }
.mono { font-family: var(--mono); }
.wrap { max-width: 1040px; margin: 0 auto; }
.eyebrow { font-family: var(--mono); font-size: 12px; letter-spacing: .12em; text-transform: uppercase; color: var(--accent); font-weight: 600; margin: 0 0 10px; }
h1 { font-size: clamp(26px,4.2vw,38px); font-weight: 700; margin: 0 0 14px; line-height: 1.14; text-wrap: balance; letter-spacing: -.01em; }
.lede { font-size: 16.5px; line-height: 1.6; color: var(--ink-soft); max-width: 68ch; margin: 0 0 6px; }
.lede strong { color: var(--ink); font-weight: 600; }
.provenance { font-family: var(--mono); font-size: 12.5px; color: var(--ink-soft); margin-top: 18px; padding-top: 18px; border-top: 1px solid var(--line-soft); display: flex; gap: 20px; flex-wrap: wrap; }
.provenance code { background: var(--ground-raised); padding: 1px 5px; border-radius: 4px; }

.panel { margin: 32px 0; border: 1px solid var(--line); border-radius: 12px; background: var(--chip-bg); box-shadow: var(--shadow); padding: 24px; }
.panel h2 { font-size: 18px; font-weight: 700; margin: 0 0 14px; }

.tabs { display: flex; gap: 6px; margin-bottom: 18px; background: var(--ground-raised); padding: 4px; border-radius: 8px; width: fit-content; }
.tab-btn { font-family: var(--mono); font-size: 12.5px; padding: 8px 14px; border: none; background: transparent; color: var(--ink-soft); border-radius: 6px; cursor: pointer; }
.tab-btn.active { background: var(--accent); color: var(--ground); font-weight: 600; }
.tab-pane { display: none; }
.tab-pane.active { display: block; }

textarea, input[type=text] { font-family: var(--mono); font-size: 13px; width: 100%; padding: 10px 12px; border-radius: 8px; border: 1px solid var(--line); background: var(--ground-raised); color: var(--ink); }
textarea { min-height: 100px; resize: vertical; }
label.field-label { font-family: var(--mono); font-size: 11px; letter-spacing: .05em; text-transform: uppercase; color: var(--ink-soft); display: block; margin: 12px 0 6px; }
label.field-label:first-child { margin-top: 0; }

.actions { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 16px; }
button { font-family: var(--mono); font-size: 13px; letter-spacing: .02em; padding: 10px 16px; border-radius: 8px; border: 1px solid var(--line); background: var(--ground-raised); color: var(--ink); cursor: pointer; transition: border-color .15s ease; }
button:hover:not(:disabled) { border-color: var(--accent); }
button.primary { background: var(--accent); border-color: var(--accent); color: var(--ground); font-weight: 600; }
button.primary:hover:not(:disabled) { opacity: .92; }
button:disabled { opacity: .5; cursor: not-allowed; }

.dropzone { border: 2px dashed var(--line); border-radius: 10px; padding: 24px 20px; text-align: center; position: relative; margin-top: 10px; }
.dropzone p { font-size: 13.5px; color: var(--ink-soft); margin: 0; }
#fileInput { position: absolute; inset: 0; opacity: 0; cursor: pointer; }

.status { font-family: var(--mono); font-size: 13px; color: var(--ink-soft); margin: 14px 0; display: none; align-items: center; gap: 10px; }
.status.show { display: flex; }
.spinner { width: 16px; height: 16px; border: 2px solid var(--line); border-top-color: var(--accent); border-radius: 50%; animation: spin .8s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }

.error-box { font-family: var(--mono); font-size: 13px; color: var(--red); background: var(--red-soft); border: 1px solid var(--red); border-radius: 8px; padding: 12px 14px; margin: 14px 0; display: none; }
.error-box.show { display: block; }

.result-card { border: 1px solid var(--line); border-radius: 10px; margin-top: 14px; overflow: hidden; background: var(--chip-bg); }
.result-card.escalate { box-shadow: inset 3px 0 0 var(--red); }
.result-head { padding: 14px 18px; border-bottom: 1px solid var(--line-soft); display: flex; justify-content: space-between; align-items: flex-start; gap: 12px; flex-wrap: wrap; }
.result-subject { font-weight: 600; font-size: 14.5px; }
.result-body-text { font-size: 13px; color: var(--ink-soft); margin-top: 4px; max-width: 60ch; }
.chips { display: flex; gap: 6px; flex-wrap: wrap; }
.chip { font-family: var(--mono); font-size: 10.5px; padding: 3px 9px; border-radius: 999px; font-weight: 600; white-space: nowrap; }
.chip.category { background: var(--accent-soft); color: var(--accent); }
.chip.sentiment-angry, .chip.sentiment-frustrated { background: var(--red-soft); color: var(--red); }
.chip.sentiment-happy { background: var(--accent-soft); color: var(--accent); }
.chip.sentiment-neutral { background: var(--ground-raised); color: var(--ink-soft); }
.chip.escalate { background: var(--red); color: var(--ground); }
.result-body { padding: 16px 18px; }
.result-body .label { font-family: var(--mono); font-size: 10.5px; letter-spacing: .06em; text-transform: uppercase; color: var(--ink-soft); margin: 0 0 6px; }
.reply-text { font-size: 14px; line-height: 1.6; white-space: pre-wrap; margin: 0 0 12px; }
.sources { font-family: var(--mono); font-size: 11px; color: var(--ink-soft); }
.sources code { background: var(--ground-raised); padding: 1px 5px; border-radius: 4px; }

.footnote { margin-top: 36px; font-size: 13px; color: var(--ink-soft); line-height: 1.6; max-width: 68ch; padding-top: 20px; border-top: 1px solid var(--line-soft); }
.footnote code { font-family: var(--mono); font-size: 12px; background: var(--ground-raised); padding: 1px 5px; border-radius: 4px; }
</style>
</head>
<body>
<div class="wrap">

  <p class="eyebrow">Customer support triage — live agent</p>
  <h1>Reads a ticket, checks the actual policy, drafts the reply.</h1>
  <p class="lede">
    Paste a customer email or upload a batch of tickets. This classifies each one's
    category and sentiment, retrieves the relevant company policy passage from a
    local knowledge base (RAG), and drafts a reply grounded in that policy —
    not invented from thin air.
  </p>
  <div class="provenance">
    <span>LLM: Groq-hosted Llama 3.1 (free tier)</span>
    <span>Retrieval: local ChromaDB vector store</span>
    <span>Policy docs are synthetic — no real company/customer data</span>
  </div>

  <div class="panel">
    <h2>Try it</h2>
    <div class="tabs">
      <button class="tab-btn active" id="tabSingleBtn" type="button">Single ticket</button>
      <button class="tab-btn" id="tabBatchBtn" type="button">Batch (CSV)</button>
    </div>

    <div class="tab-pane active" id="tabSingle">
      <label class="field-label" for="singleSubject">Subject</label>
      <input type="text" id="singleSubject" placeholder="e.g. Package never arrived">
      <label class="field-label" for="singleBody">Ticket body</label>
      <textarea id="singleBody" placeholder="Paste the customer's email text here..."></textarea>
      <div class="actions">
        <button type="button" class="primary" id="btnRunSingle">Run triage</button>
        <button type="button" id="btnLoadExample">Load an example</button>
      </div>
    </div>

    <div class="tab-pane" id="tabBatch">
      <div class="actions" style="margin-top:0">
        <button type="button" class="primary" id="btnRunSample">Try the sample batch (10 tickets)</button>
      </div>
      <div class="dropzone">
        <p>Or drop a CSV here (must have "subject" and "body" columns)</p>
        <input type="file" id="fileInput" accept=".csv">
      </div>
    </div>

    <div class="status" id="status"><div class="spinner"></div><span id="statusMsg">Running...</span></div>
    <div class="error-box" id="errorBox"></div>
  </div>

  <div id="resultsList"></div>

  <p class="footnote">
    Every reply is grounded in the retrieved policy passage shown under "policy used" —
    the model is explicitly instructed not to invent policy details it wasn't given.
    Tickets flagged for escalation (safety issues, threatened legal action, or
    repeated failed resolution) get a red marker rather than an auto-drafted reply
    being treated as final.
  </p>

</div>

<script>
const tabSingleBtn = document.getElementById('tabSingleBtn');
const tabBatchBtn = document.getElementById('tabBatchBtn');
const tabSingle = document.getElementById('tabSingle');
const tabBatch = document.getElementById('tabBatch');
const status = document.getElementById('status');
const statusMsg = document.getElementById('statusMsg');
const errorBox = document.getElementById('errorBox');
const resultsList = document.getElementById('resultsList');

tabSingleBtn.addEventListener('click', () => {
  tabSingleBtn.classList.add('active'); tabBatchBtn.classList.remove('active');
  tabSingle.classList.add('active'); tabBatch.classList.remove('active');
});
tabBatchBtn.addEventListener('click', () => {
  tabBatchBtn.classList.add('active'); tabSingleBtn.classList.remove('active');
  tabBatch.classList.add('active'); tabSingle.classList.remove('active');
});

function showError(msg) { errorBox.textContent = msg; errorBox.classList.add('show'); }
function clearError() { errorBox.classList.remove('show'); errorBox.textContent = ''; }
function setBusy(busy, msg) {
  status.classList.toggle('show', busy);
  if (msg) statusMsg.textContent = msg;
  document.querySelectorAll('button').forEach(b => b.disabled = busy);
}

function sentimentClass(s) { return 'sentiment-' + (s || 'neutral').toLowerCase(); }

function renderResult(r) {
  const escalateCls = r.needs_escalation ? ' escalate' : '';
  const escalateChip = r.needs_escalation
    ? `<span class="chip escalate">ESCALATE</span>` : '';
  const sources = (r.policy_sources || []).map(s => `<code>${s.replace(/_/g, ' ')}</code>`).join(' ');
  return `
    <div class="result-card${escalateCls}">
      <div class="result-head">
        <div>
          <div class="result-subject">${r.subject ? escapeHtml(r.subject) : '(no subject)'}</div>
          ${r.body ? `<div class="result-body-text">${escapeHtml(r.body.slice(0, 140))}${r.body.length > 140 ? '…' : ''}</div>` : ''}
        </div>
        <div class="chips">
          <span class="chip category">${escapeHtml(r.category)}</span>
          <span class="chip ${sentimentClass(r.sentiment)}">${escapeHtml(r.sentiment)}</span>
          ${escalateChip}
        </div>
      </div>
      <div class="result-body">
        <p class="label">Suggested reply</p>
        <p class="reply-text">${escapeHtml(r.suggested_reply)}</p>
        <p class="sources">Policy used: ${sources || 'none retrieved'}</p>
        ${r.needs_escalation ? `<p class="sources" style="color:var(--red);margin-top:6px">Escalation reason: ${escapeHtml(r.escalation_reason || '')}</p>` : ''}
      </div>
    </div>
  `;
}

function escapeHtml(s) {
  const d = document.createElement('div');
  d.textContent = s == null ? '' : String(s);
  return d.innerHTML;
}

document.getElementById('btnLoadExample').addEventListener('click', () => {
  document.getElementById('singleSubject').value = "STILL NOT WORKING after 3 tries!!";
  document.getElementById('singleBody').value = "This is the THIRD time I've reset my Northbridge Connect smart plug and it still won't connect to my wifi. I've followed every step in your app. I am extremely frustrated and considering returning everything I've bought from you.";
});

document.getElementById('btnRunSingle').addEventListener('click', async () => {
  clearError(); resultsList.innerHTML = '';
  const subject = document.getElementById('singleSubject').value.trim();
  const bodyText = document.getElementById('singleBody').value.trim();
  if (!bodyText) { showError('Please enter a ticket body.'); return; }
  setBusy(true, 'Classifying, retrieving policy, drafting reply...');
  try {
    const res = await fetch('/api/triage-one', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ subject, body: bodyText }),
    });
    const data = await res.json();
    if (!res.ok) { showError(data.error || 'Something went wrong.'); return; }
    resultsList.innerHTML = renderResult({ subject, body: bodyText, ...data.result });
  } catch (e) {
    showError('Network error — please try again.');
  } finally { setBusy(false); }
});

async function runBatch(formData) {
  clearError(); resultsList.innerHTML = '';
  setBusy(true, 'Processing batch — this runs one LLM call per ticket, please wait...');
  try {
    const res = await fetch('/api/triage-batch', { method: 'POST', body: formData });
    const data = await res.json();
    if (!res.ok) { showError(data.error || 'Something went wrong.'); return; }
    resultsList.innerHTML = data.results.map(renderResult).join('');
    if (data.truncated) showError('Note: batch was truncated to the first 50 rows for this demo.');
  } catch (e) {
    showError('Network error — please try again.');
  } finally { setBusy(false); }
}

document.getElementById('btnRunSample').addEventListener('click', () => {
  const fd = new FormData(); fd.append('use_sample', '1'); runBatch(fd);
});
document.getElementById('fileInput').addEventListener('change', (e) => {
  if (!e.target.files.length) return;
  const fd = new FormData(); fd.append('file', e.target.files[0]); runBatch(fd);
});
</script>
</body>
</html>
"""

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), threaded=True)
