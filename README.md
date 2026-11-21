# filing_analyst

Ask questions about public company annual reports (SEC Form 10-K) and get short
answers that cite the exact passages they came from. A retrieval augmented
generation (RAG) project: it first finds the right passages, then has a language
model write an answer from only those, then checks the answer with plain code.

```
$ python -m filing_analyst ask "What was Apple's total revenue for fiscal 2023?"
Apple's total net sales were $383,285 million in fiscal 2023 [1].

[1] Apple Inc. 2023 10-K, Item 7 Management's Discussion and Analysis
```

(That output is an illustration of the format. Real results depend on the filings
you index.)

## How it works

```
10-K HTML file
   |  parse.py      strip hidden data, rebuild tables as text rows, split by "Item N"
   v
sections
   |  chunk.py      passages of up to 300 words, tables kept together, source labels attached
   v
passages -----> store.py   SQLite keyword index (FTS5)  +  vectors in one numpy file
                                   |
question -> retrieve.py: read company and year from the question
                         keyword search + vector search, merged (reciprocal rank fusion)
                         reranker reorders the best 30            -> top 5 passages
                                   |
                           answer.py: model answers from the passages, citing [1] [2]
                                      code then checks every citation and every number
```

Choices that matter, and why:

- **Section aware parsing.** EDGAR files are messy HTML: a table of contents that
  looks like real headings, words split across spans, hidden XBRL data, and tables
  that put `$` and `)` in separate cells. The parser handles these, and each
  passage remembers its company, year and Item so answers can cite it.
- **Company and year filters read from the question.** With many companies indexed,
  "What was Apple's revenue?" is easily answered with another company's passage
  that uses the same words. If the question names a company or a year, the search
  is restricted to it. A question naming two companies searches each and merges.
- **Keyword plus meaning search.** Keyword search is exact on names, numbers and
  jargon; vector search finds paraphrases ("power cell vendors" for "battery
  suppliers"). Reciprocal rank fusion merges the two lists without having to
  compare their incompatible scores.
- **A reranker** (a small cross encoder that reads question and passage together)
  re orders the top 30. It is slower but better at picking the single best passage.
- **Answers are checked, not trusted.** After the model answers, code verifies that
  each citation points to a real passage and that every number in the answer
  appears in a passage the answer cites. An answer that fails is printed with a
  warning. If nothing relevant is found, the model is told to say so instead of
  guessing.
- **Few moving parts.** SQLite and numpy instead of a vector database service; the
  embedding and reranker models run on a laptop CPU.

## Setup

```
pip install -e ".[dev]"
export SEC_USER_AGENT="Your Name you@example.com"    # the SEC requires this
python scripts/download_filings.py data AAPL MSFT NVDA GOOGL AMZN --years 2
python -m filing_analyst ingest data
python -m filing_analyst search "supply chain risks at Nvidia"     # no API key needed
export ANTHROPIC_API_KEY=...                                       # only for ask
python -m filing_analyst ask "What was Microsoft's net income for fiscal 2024?"
```

The first run downloads the two small models (about 200 MB total).

## Web API

The same search and answers are available over HTTP, so a web page, a chat bot
or another program can use them.

```
pip install -e ".[api]"
uvicorn filing_analyst.api:app --port 8000
```

Then open http://localhost:8000/docs for an interactive page where every endpoint
can be tried from the browser.

| Endpoint | What it does |
|---|---|
| `GET /health` | Is the server up, how many passages are loaded, is the reranker loaded, is `/ask` on |
| `POST /search` | The best matching passages for a question. Free, no API key needed |
| `POST /ask` | A written answer with checked citations. Needs `ANTHROPIC_API_KEY`; without it, returns 503 |

```
$ curl -s -X POST localhost:8000/search -H 'Content-Type: application/json' \
       -d '{"question": "What was NVIDIA total revenue for fiscal 2026?", "k": 3}'
{
  "question": "What was NVIDIA total revenue for fiscal 2026?",
  "filters": {"tickers": ["NVDA"], "year": 2026},
  "reranked": true,
  "results": [
    {"rank": 1, "label": "NVIDIA CORP 2026 10-K, Item 7 Management's Discussion and Analysis ...",
     "ticker": "NVDA", "fiscal_year": 2026, "item": "7", "kind": "text", "text": "...", "score": 8.4},
    ...
  ]
}
```

Request fields: `question` (1 to 1,000 characters), `k` (1 to 20, default 5), and
for `/search` also `mode` (`hybrid` default, `dense` or `bm25`), `rerank` (default
true) and `use_filters` (default true). Anything else is rejected with a 422 that
says what is wrong, before any search runs or any model is called.

**Choosing a mode.** The default is the full pipeline (hybrid search, company and
year filter, reranker), the same as the command line. On the number lookup
questions in the eval below, vector search alone did better and was much faster
(about 0.05 s against 1.3 s per request on a laptop), so for questions like "What
was X's revenue in year Y" send `"mode": "dense", "rerank": false`. The eval does
not cover paraphrased or wording based questions, which is where keyword search
and the reranker are meant to help, so the default was not changed on its evidence.

Settings, read when the server starts:

| Variable | Default | Meaning |
|---|---|---|
| `FILING_ANALYST_INDEX` | `index` | index folder made by `ingest` |
| `FILING_ANALYST_RERANK` | `1` | `0` skips loading the reranker (faster start, less memory) |
| `ANTHROPIC_API_KEY` | unset | turns on `/ask` |

How it behaves under load and errors:

- The index and models load once at startup. If the index is empty the server
  refuses to start and says to run `ingest` first.
- Requests run on several threads, and the shared database connection and models
  are used by one search at a time (a lock). Without the lock, parallel requests
  corrupt each other; a test sends 24 requests at once to check this. The slow
  language model call in `/ask` runs outside the lock, so it does not hold up
  other searches.
- If the language model fails, `/ask` returns 502 with a short message. The full
  error goes only to the server log, because it can contain keys or internal
  details.

## Measuring it

Claims about search quality need numbers, so there is a scoring tool.

```
python scripts/make_eval_questions.py data eval/questions.jsonl
python -m filing_analyst eval eval/questions.jsonl
```

`make_eval_questions.py` builds questions from the SEC's own structured financial
data (revenue, net income, operating income and total assets for each filing), so
the right answers do not come from a language model. A search counts as correct if
one of the top 5 passages comes from the right company and year and contains the
number. The eval then compares five setups, so each design choice has to earn its
place:

1. keyword search only
2. vector search only
3. both merged
4. both merged plus the company and year filter
5. all of that plus the reranker

It reports recall@5 (how often the right passage was in the top 5) and MRR
(how high it ranked). Questions whose answer text is nowhere in the filing, for
example because the filing writes the number differently, are left out and counted
rather than scored.

### Results

Not filled in yet: this needs real filings, which have to be downloaded from the
SEC on your own machine. Run the commands above and paste the table here.

## Tests

`pytest` runs 70+ tests, no network needed except for the model download:

- parser: table of contents skipped, hidden data removed, table cells cleaned,
  words split across tags kept whole, cross references not mistaken for headings
- chunking: size limits, long paragraphs split at sentences, large tables split with
  their caption
- search: safe handling of odd characters in questions, filters, rank fusion math,
  vector row to database row alignment, multi company questions
- answers: a correct answer passes; an invented number, a number taken from an
  uncited passage, a made up citation and a missing citation are each caught
- scoring: metrics computed by hand on small cases
- web API: every endpoint, ten kinds of bad input, broken JSON, search syntax in
  the question, 24 parallel requests, a model that invents a number, refuses or
  fails, error messages that must not leak secrets, and no model call (no cost)
  for a rejected request
- real models: a paraphrase that shares no words with the passage is still found

The sample filings in `tests/fixtures.py` are **fictional** companies with
invented numbers. They imitate the awkward parts of real EDGAR files but are not
real data.

## Limits

- Only Form 10-K. No 10-Q quarterlies, no earnings call transcripts.
- "Fiscal year" in file names and filters means the calendar year in which the
  company's fiscal year ended, so a year ending January 2024 is 2024.
- Tables are flattened to text rows, so a question that depends on reading a
  complicated multi column table can fail.
- The number check catches invented numbers, not wrong reasoning, such as
  comparing two correct numbers the wrong way.
- Question text is only matched to a company by name or ticker, so a vague
  reference like "the iPhone maker" will not trigger the filter.
