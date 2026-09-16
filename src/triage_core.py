"""
triage_core.py
==============
The actual triage logic: classify a support ticket, judge sentiment,
retrieve the relevant policy passages (RAG via a local ChromaDB vector
store), and draft a suggested reply grounded in those passages.

No web framework code here — this module is pure logic, importable and
testable on its own.
"""
import os
import json
import glob
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from groq import Groq

POLICIES_DIR = Path(os.environ.get("POLICIES_DIR", Path(__file__).resolve().parent.parent / "policies"))
CHROMA_DIR = Path(os.environ.get("CHROMA_DIR", Path(__file__).resolve().parent.parent / ".chroma"))
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")

CATEGORIES = [
    "Refund Request",
    "Technical Support",
    "Sales Inquiry",
    "Shipping Issue",
    "Complaint / Escalation",
    "General Question",
]

_client = None
_collection = None
_groq_client = None


def _get_groq_client():
    global _groq_client
    if _groq_client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        _groq_client = Groq(api_key=api_key)
    return _groq_client


def _chunk_markdown(text, chunk_size=600, overlap=80):
    """Simple paragraph-aware chunking: split on blank lines, then pack
    paragraphs into ~chunk_size-character windows with a little overlap so
    a policy point split across a chunk boundary isn't lost."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for p in paragraphs:
        if len(current) + len(p) + 2 > chunk_size and current:
            chunks.append(current.strip())
            # keep a small tail of the previous chunk for context continuity
            current = current[-overlap:] + "\n\n" + p
        else:
            current = (current + "\n\n" + p).strip() if current else p
    if current.strip():
        chunks.append(current.strip())
    return chunks


def build_or_load_vector_store(force_rebuild=False):
    """Loads policy markdown files into a local ChromaDB collection,
    embedding each chunk with a small local sentence-transformer model
    (no API calls needed for embeddings — only the LLM calls use Groq)."""
    global _client, _collection

    _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    existing = [c.name for c in _client.list_collections()]
    if "policies" in existing and not force_rebuild:
        _collection = _client.get_collection("policies", embedding_function=embed_fn)
        return _collection

    if "policies" in existing:
        _client.delete_collection("policies")
    _collection = _client.create_collection("policies", embedding_function=embed_fn)

    ids, docs, metadatas = [], [], []
    for filepath in sorted(glob.glob(str(POLICIES_DIR / "*.md"))):
        source_name = Path(filepath).stem
        text = Path(filepath).read_text(encoding="utf-8")
        for i, chunk in enumerate(_chunk_markdown(text)):
            ids.append(f"{source_name}-{i}")
            docs.append(chunk)
            metadatas.append({"source": source_name})

    if docs:
        _collection.add(ids=ids, documents=docs, metadatas=metadatas)
    return _collection


def retrieve_policy_context(query_text, n_results=3):
    """Returns the top-N most relevant policy chunks for a ticket's text."""
    if _collection is None:
        build_or_load_vector_store()
    results = _collection.query(query_texts=[query_text], n_results=n_results)
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    return [
        {"source": m.get("source", "unknown"), "text": d}
        for d, m in zip(docs, metas)
    ]


def classify_ticket(subject, body):
    """One Groq call: category + sentiment, forced JSON output."""
    client = _get_groq_client()
    prompt = f"""Classify this customer support ticket.

Subject: {subject}
Body: {body}

Categories (pick exactly one): {", ".join(CATEGORIES)}

Sentiment (pick exactly one): "angry", "frustrated", "neutral", "happy"

Also flag if this ticket needs urgent human escalation (true only for
safety issues, threats of legal action, or a customer who is extremely
angry after repeated failed attempts at resolution — not for routine
complaints).

Return EXACTLY this JSON, no extra text:
{{
  "category": "...",
  "sentiment": "...",
  "needs_escalation": true or false,
  "escalation_reason": "brief reason, or empty string if not escalating"
}}"""
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": "You are a precise support-ticket triage classifier. Return only valid JSON."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=800,
    )
    # gpt-oss models are reasoning models: max_tokens covers the internal
    # "reasoning" field too, so a low budget can leave `content` empty even
    # though the call succeeds (finish_reason="stop", not "length" — this
    # is a genuine gotcha with this model family, not an error to swallow
    # silently). Fall back to the reasoning text as a last resort so a
    # thin budget degrades gracefully instead of always hitting the
    # hardcoded default.
    msg = resp.choices[0].message
    text = (msg.content or "").strip()
    if not text and getattr(msg, "reasoning", None):
        text = msg.reasoning.strip()
    return _safe_json(text, fallback={
        "category": "General Question", "sentiment": "neutral",
        "needs_escalation": False, "escalation_reason": "",
    })


def draft_reply(subject, body, category, sentiment, policy_context):
    """Second Groq call: draft a reply grounded in the retrieved policy
    passages. The model is told explicitly to only use the provided
    context, not invent policy details."""
    client = _get_groq_client()
    context_block = "\n\n".join(
        f"[From {c['source'].replace('_', ' ')}]\n{c['text']}" for c in policy_context
    ) or "No specific policy passage was retrieved for this ticket."

    tone_hint = {
        "angry": "The customer is angry. Open with a genuine, specific acknowledgment of their frustration before anything else — not a generic 'we're sorry for the inconvenience.'",
        "frustrated": "The customer is frustrated. Be direct and solution-first; avoid over-apologizing.",
        "happy": "The customer is in a good mood. Keep the reply warm but still efficient — don't over-explain.",
        "neutral": "Keep the tone friendly and straightforward.",
    }.get(sentiment, "Keep the tone friendly and professional.")

    prompt = f"""Draft a customer support reply.

Customer's ticket — Subject: {subject}
Body: {body}

Category: {category}
Customer sentiment: {sentiment}
{tone_hint}

RELEVANT COMPANY POLICY (use only this — do not invent policy details
not stated here; if the policy doesn't cover their specific question,
say a specialist will follow up rather than guessing):
{context_block}

Write a short, ready-to-send reply (3-6 sentences). Sign off as "The Northbridge Support Team"."""
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": "You write clear, grounded customer support replies. Never invent policy details not given to you."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
        max_tokens=900,
    )
    msg = resp.choices[0].message
    text = (msg.content or "").strip()
    if not text and getattr(msg, "reasoning", None):
        text = msg.reasoning.strip()
    return text


def triage_one(subject, body):
    """Full pipeline for one ticket: classify -> retrieve -> draft."""
    classification = classify_ticket(subject, body)
    policy_context = retrieve_policy_context(f"{subject}\n{body}", n_results=3)
    reply = draft_reply(
        subject, body,
        classification["category"], classification["sentiment"],
        policy_context,
    )
    return {
        **classification,
        "policy_sources": [c["source"] for c in policy_context],
        "suggested_reply": reply,
    }


def _safe_json(text, fallback):
    try:
        return json.loads(text)
    except Exception:
        import re
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
        return fallback
