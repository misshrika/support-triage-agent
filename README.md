# Support Triage Agent

**The problem:** a support inbox fills up with hundreds of emails — refund
requests, angry complaints, technical issues, sales inquiries — all mixed
together. Someone has to read each one, figure out what kind of request it
is, judge how upset the customer is, look up the relevant policy, and write
a reply. That's slow, repetitive, and the quality of the reply depends on
which agent happens to pick up the ticket and how well they remember policy
details.

**What changed:** this agent reads each ticket, classifies its category and
sentiment, retrieves the actual relevant policy passage from a local
knowledge base, and drafts a reply grounded in that retrieved policy —
turning "read, categorize, look up policy, write reply" into a few seconds
of automated first-pass work that a human still reviews before it's sent.

---

## How it works (RAG — Retrieval-Augmented Generation)

1. **Classify.** One LLM call reads the ticket's subject and body and
   returns a structured category (Refund Request, Technical Support, Sales
   Inquiry, Shipping Issue, Complaint/Escalation, General Question) and
   sentiment (angry, frustrated, neutral, happy), plus an escalation flag
   for tickets that genuinely need a human immediately (safety issues,
   legal threats, repeated failed resolution) rather than an auto-drafted
   reply.
2. **Retrieve.** The ticket's text is embedded and matched against a local
   vector store (ChromaDB) built from the company's own policy documents —
   this returns the specific policy passages actually relevant to *this*
   ticket, not the whole policy document.
3. **Draft.** A second LLM call writes a reply, given only the retrieved
   policy passages as its source of truth. The prompt explicitly forbids
   inventing policy details that weren't retrieved — if the knowledge base
   doesn't cover the customer's specific question, the reply says a
   specialist will follow up, rather than guessing.

## The engineering worth reading

- **Retrieval grounds the model, rather than trusting its memory.** A
  general-purpose LLM asked "what's our refund policy" will happily
  hallucinate a plausible-sounding but wrong answer. Retrieving the actual
  policy text first and instructing the model to use only that text is
  what makes the drafted reply trustworthy enough to review quickly rather
  than fact-check line by line.
- **Two separate LLM calls, not one.** Classification and drafting are
  split into separate calls with different temperatures — classification
  uses `temperature=0` for consistent, repeatable categorization; drafting
  uses a higher temperature for more natural-sounding prose. Bundling both
  into one call makes each task worse at the other's expense.
- **Escalation is judged conservatively.** The classifier is explicitly
  told to reserve the escalation flag for safety issues, legal threats, or
  a customer already through multiple failed resolution attempts — not
  every complaint or angry email. A system that escalates too eagerly is
  as useless as one that never does.
- **Sentiment shapes tone, not just a label.** An angry-ticket reply is
  instructed to open with a specific acknowledgment of the actual
  frustration rather than a generic "sorry for the inconvenience" —  a
  small prompt-design detail that meaningfully changes how a drafted reply
  reads.
- **Chunking preserves context across boundaries.** Policy documents are
  split into overlapping chunks (not naive fixed-length slices) so a
  policy point that would otherwise land right at a chunk boundary doesn't
  get silently cut in half during retrieval.

## Run it

```powershell
pip install -r requirements.txt
# Get a free API key at https://console.groq.com -> API Keys
# copy .env.example to .env and add it
python src/app.py
```

Open `http://localhost:5000`. Try a single pasted ticket, or run the
10-ticket sample batch.

## Tech stack

Python · Flask · Groq API (free-tier hosted Llama 3.1 — classification and
reply drafting) · ChromaDB (local vector store) · `sentence-transformers`
(local embedding model — no API cost for retrieval) · pandas (batch CSV
processing)

## Note

All policy documents and sample tickets in this repo are synthetic — a
fictional company ("Northbridge Home Goods"), invented for demonstration.
No real customer data or company policy is used anywhere in this repo.
