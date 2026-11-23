# filing_analyst

[![CI](https://github.com/bcherukuri2004/filing_analyst/actions/workflows/ci.yml/badge.svg)](https://github.com/bcherukuri2004/filing_analyst/actions/workflows/ci.yml)

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
| `GET /health` | Is the server up, can it reach the database (`"database": "reachable"`), how many passages are loaded, is the reranker loaded, is `/ask` on, is a key required. Answers 503 with `"database": "unreachable"` while the database is down. Always open, so monitoring can reach it |
| `POST /search` | The best matching passages for a question. No model cost |
| `POST /ask` | A written answer with checked citations. Needs `ANTHROPIC_API_KEY`; without it, returns 503. Rate limited per client and in total |

If the server has a key set (see "Protecting it" below), `/search` and `/ask` need
it in an `X-API-Key` header: add `-H "X-API-Key: $FILING_ANALYST_API_KEY"` to the
example below.

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
| `FILING_ANALYST_INDEX` | `index` | index folder made by `ingest`, or a `postgresql://` address (see "Using Postgres") |
| `FILING_ANALYST_RERANK` | `1` | `0` skips loading the reranker (faster start, less memory) |
| `ANTHROPIC_API_KEY` | unset | turns on `/ask` |
| `FILING_ANALYST_API_KEY` | unset | if set, `/search` and `/ask` require it in an `X-API-Key` header. At least 16 characters |
| `FILING_ANALYST_ASK_PER_MINUTE` | `10` | how many `/ask` calls each client may make per minute. A whole number, 1 or more |
| `FILING_ANALYST_ASK_GLOBAL_PER_MINUTE` | `60` | how many `/ask` calls everyone together may make per minute. This is the ceiling on model spending. A whole number, 1 or more |

A setting with a bad value (a key shorter than 16 characters, a limit of `0` or
`abc`) stops the server at startup with a message saying what is wrong, rather than
running with protection quietly off. No value means "no limit": both limits are
always on. An empty `FILING_ANALYST_API_KEY` counts as
unset.

### Protecting it

`/ask` calls a paid language model, so every call costs money. Out of the box
`uvicorn` only listens on `127.0.0.1`, which means only your own machine can reach
the server. **Before making it reachable from other machines** (`--host 0.0.0.0`,
a container, a cloud server), set a key:

```
export FILING_ANALYST_API_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(24))')"
uvicorn filing_analyst.api:app --host 0.0.0.0 --port 8000
curl -s -X POST localhost:8000/search -H 'Content-Type: application/json' \
     -H "X-API-Key: $FILING_ANALYST_API_KEY" -d '{"question": "Apple revenue 2026"}'
```

| Reply | Meaning |
|---|---|
| 401 | No key, or the wrong key. The reply never contains the real key |
| 429 | Too many `/ask` calls in the last minute, either from this client or from everyone together (the message says which). The `Retry-After` header says how many seconds to wait |

Details:

- Keys are compared with `hmac.compare_digest`, which takes the same time however
  much of a guess is right, so a key cannot be worked out by timing replies.
- Requests rejected for a missing key or bad input never reach the model and do
  not count towards the limit, so nobody can use up another person's allowance
  without the key.
- There are two limits on `/ask`. **Per client** (default 10 a minute) stops one
  caller from using more than their share. **For everyone together** (default 60
  a minute) puts a ceiling on spending: a per-client limit alone cannot, because
  someone with 1,000 addresses would get 1,000 allowances. With the defaults,
  `/ask` can never make more than 60 model calls a minute, however many
  addresses are used.
- The two limits are checked fairly. A client's own limit is checked first, so a
  client that is over it, and keeps trying, never uses up the shared allowance
  and cannot lock everyone else out. A client turned away because the server is
  busy is not charged for that try.
- A client means an IP address, with one exception: IPv6 addresses in the same
  `/64` network count as one client. A home or phone connection is usually given
  a whole `/64` (billions of addresses), so counting each address separately
  would let one person look like billions of clients. An IPv4 address written in
  IPv6 form (`::ffff:10.0.0.1`) counts as that IPv4 address.
- Both limits are kept in memory: they reset when the server restarts, and each
  server process counts separately, so 4 processes allow up to 4 x 60 a minute.
  Behind a proxy or load balancer every request comes from the proxy's address,
  so all users would share one per-client limit (the overall limit still holds).
  For that setup, put the per-client limit in the proxy instead.
- Memory for the limit has a hard cap: it remembers at most 10,000 clients. When
  a new address arrives and the table is full, the client seen longest ago is
  forgotten, and every call does the same small amount of work however full the
  table is. (An earlier version had no real cap: a flood of 50,000 addresses,
  cheap to get with IPv6, grew the table to 50,000 entries and made each call
  about 5,000 times slower. A test now floods it to check this.)
- Trade-off of the cap: if a flood of new addresses fills the table, a client
  that has gone quiet can be forgotten and get a fresh limit. A client that keeps
  calling, even one being refused with 429, counts as recently seen, so a flood
  cannot reset the limit of someone who is actively hitting it.

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

## Running in Docker

The container holds the code, its packages and both models, so it runs the same
on any machine with Docker and needs no Python setup. It does not hold the search
index or any keys: the index is mounted from your machine read-only, and keys are
passed in when it starts.

Build the index once (see Setup), then:

```
docker compose up --build -d          # or: docker-compose up --build -d
curl -s localhost:8000/health
docker compose logs -f                # watch it
docker compose down                   # stop and remove it
```

The first build takes a few minutes (packages and about 150 MB of models). Later
builds reuse those steps and take seconds unless `pyproject.toml` changes.

**Keys and settings.** Put them in a file named `.env` next to
`docker-compose.yml`, which Compose reads automatically and git ignores:

```
FILING_ANALYST_API_KEY=<a long random key, see "Protecting it">
ANTHROPIC_API_KEY=<only if you want /ask>
FILING_ANALYST_ASK_PER_MINUTE=10
FILING_ANALYST_ASK_GLOBAL_PER_MINUTE=60
```

Also available: `FILING_ANALYST_PORT` (default 8000), `FILING_ANALYST_INDEX_DIR`
(default `./index`) and `FILING_ANALYST_RERANK`. Keys are never written into the
image, so the image can be shared without sharing them.

**Updating the data.** Run `ingest` again on your machine, then
`docker compose restart`. No rebuild is needed, because the index is not part of
the image.

How the container is locked down:

| Setting | What it means |
|---|---|
| Port published as `127.0.0.1:8000` | Only this machine can connect. Change it only after setting `FILING_ANALYST_API_KEY` |
| Index mounted `:ro` | The app cannot change or damage the index |
| Runs as user `app` (uid 10001), not root | A break-in into the app gets very little |
| `read_only: true`, only `/tmp` writable | The app cannot change its own files |
| `no-new-privileges`, `cap_drop: ALL` | It cannot gain extra powers after starting |
| `HF_HUB_OFFLINE=1`, models inside the image | It never downloads anything while running |
| Health check on `/health` | `docker ps` shows `healthy`, not just `running` |

The image is about 870 MB, mostly the model runtime (onnxruntime), numpy and the
models themselves.

## Using Postgres instead of SQLite

SQLite is the default because it needs no setup: the index is a folder. For a
shared database server there is a second store, **Postgres with the pgvector
extension**. Everything else (search, answers, the eval, the API) works the same
on either, because both stores have the same methods. One setting picks the store:
a folder means SQLite, and a `postgresql://` address means Postgres.

```
pip install -e ".[postgres]"
echo "POSTGRES_PASSWORD=$(python -c 'import secrets; print(secrets.token_urlsafe(18))')" >> .env
docker compose --profile postgres up -d db        # Postgres 17 + pgvector, on 127.0.0.1:5433

export PG="postgresql://filing:<the password>@127.0.0.1:5433/filing"
python -m filing_analyst --index "$PG" ingest data
python -m filing_analyst --index "$PG" search "NVIDIA revenue 2026"
FILING_ANALYST_INDEX="$PG" uvicorn filing_analyst.api:app --port 8000
```

The database uses port **5433**, not the usual 5432, so it cannot clash with a
Postgres already installed on your machine. To run the API container against it,
set `FILING_ANALYST_CONTAINER_INDEX=postgresql://filing:<password>@db:5432/filing`
in `.env` (inside Docker the database is called `db`) and run
`docker compose --profile postgres up -d`. Without `--profile postgres` the
database is not started at all.

How the Postgres store works:

| Part | SQLite store | Postgres store |
|---|---|---|
| Keyword search | FTS5, ranked with BM25 | built in full text search (`tsvector`, GIN index), ranked with `ts_rank_cd` |
| Vector search | every vector compared in numpy | pgvector, HNSW index, inner product |
| Company and year filters | applied in Python after scoring | a `WHERE` in the same query |
| Replacing the index | one transaction | one transaction: a failure halfway leaves the old index untouched |

Details that matter:

- **Filtered vector search returns every match.** An HNSW index finds near
  neighbours first and filters afterwards, so by default a filter (or asking for
  more than 40 rows) can quietly return fewer rows than asked for. The store turns
  on pgvector's iterative scans, which keep looking. A test forces Postgres to use
  the index on 3,000 rows and checks the counts.
- **It survives a database restart.** If the connection is lost, the server
  reconnects once and retries, so a short blip goes unnoticed. While the
  database is really down, `/health` and `/search` answer 503 with a plain
  message instead of crashing, and the server recovers by itself when the
  database is back; no restart needed. Tested by stopping the real database
  container under a running server, and by cutting the connection from inside
  Postgres in the test suite.
- **Passwords never show.** Logs, messages and `/health` show the address as
  `postgresql://filing:***@...`. `/health` only says `"store": "postgres"`.
- **Bad input is refused before anything is deleted**: vectors with NaN or
  infinite values, or a different number of vectors and passages. Postgres cannot
  store the NUL character, so both stores drop it from text.
- **Equal scores come back in id order** on both stores, so the same question
  always gives the same results.

**SQLite and Postgres compared** (the same 10 filings and 40 questions as below):

| Setup | SQLite recall@5 | SQLite MRR | Postgres recall@5 | Postgres MRR |
|---|---|---|---|---|
| keyword search only | 0.000 (BM25) | 0.000 | 0.050 (ts_rank) | 0.017 |
| vector search only | 1.000 | 0.835 | 1.000 | 0.835 |
| hybrid (keyword + vector) | 0.500 | 0.212 | 0.875 | 0.527 |
| hybrid + company/year filter | 0.475 | 0.263 | 0.900 | 0.721 |
| hybrid + filter + reranker | 0.800 | 0.415 | 0.850 | 0.399 |

Vector search is identical, as it should be: same vectors, same maths. Hybrid
search scores higher on Postgres, **but not because Postgres search is better.
Its keyword search is worse.** `ts_rank_cd` gives every word the same weight,
while BM25 gives rare words (like a company name) more weight than common ones
(like "revenue"). Measured on these questions, the keyword top 10 is from the
right company 81% of the time on SQLite and 34% on Postgres. Weaker keyword
results add less competition when the two lists are merged, so the vector
search's correct answer stays near the top. The lesson is the same as in
"Results" below: on these questions, keyword search hurts, and the better fix is
to give it the company and year labels it lacks (see "Limits"), not to switch
databases.

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

### The wider eval

The first question set (above) asks 4 figures per filing in one fixed wording,
with answers fetched from the SEC's API. The wider set asks **13 kinds of
figure** (revenue, cost of revenue, gross profit, R&D, operating income, income
tax, net income, operating cash flow, capital spending, cash, total assets, total
liabilities, shareholders' equity), each in one of **three wordings**, some of
which avoid the filing's own words ("How much did Apple bring in from sales").

```
python scripts/make_wide_eval.py data eval/questions_wide.jsonl
python -m filing_analyst eval eval/questions_wide.jsonl
```

The answers come from the **inline XBRL tags inside the filings themselves**:
every official number in a 10-K is tagged with what it is and which period and
part of the company it covers. The builder only takes the whole-company figure
for the full year the filing covers, never last year's column, a single segment
or one quarter, and keeps the number exactly as printed. No network is needed.
As a cross-check, all 40 answers from the SEC's API match what the builder reads
from the files, from two independent sources. A figure a company does not tag
(Amazon reports no gross profit line) gets no question, so the 10 filings give
117 questions.

The eval also shows recall **by kind of question**, because one average can hide
a whole kind failing.

### The gate

`eval --gate eval/gate.json` turns the eval into a pass or fail check: the
command exits with an error if any score is under the minimum in the gate file,
or if fewer than `min_questions` questions could be graded. CI runs it on the
wider set.

```
python -m filing_analyst eval eval/questions_wide.jsonl --gate eval/gate.json
```

The question count matters as much as the scores. When a parser bug was put
back on purpose (headings laid out as tables, which Amazon uses), both Amazon
filings lost all their sections and their questions could no longer be graded:
8 of 40 on the first set, 19 of 117 on the wider one. On the first set, the
scores of the remaining questions went **up** (vector MRR 0.835 to 0.846). Only
the question count caught it, on both sets. The minimums sit a little under
today's scores, so a real drop fails but tiny differences between machines do
not.

### Results

Two 10-K filings each for Apple, Microsoft, NVIDIA, Alphabet and Amazon (10
filings, 2,960 passages), 40 questions:

| Setup | recall@5 | MRR |
|---|---|---|
| keyword search only (BM25) | 0.000 | 0.000 |
| vector search only | 1.000 | 0.835 |
| hybrid (keyword + vector) | 0.500 | 0.212 |
| hybrid + company/year filter | 0.475 | 0.263 |
| hybrid + filter + reranker | 0.800 | 0.415 |

Vector search alone wins. The answers sit in financial tables, and each vector is
made from the passage *plus* a label with the company and year, while keyword
search only sees the passage. So keyword search finds revenue tables for the wrong
company or year, and merging its results in pushes the right answer down. This is
a design finding, not a bug, and it is why the API lets you choose `"mode":
"dense"`. The obvious next step is to add the same labels to the keyword index.

**The wider set** (117 questions, 13 kinds of figure, varied wording) is harder,
and shows what the first set hid:

| Setup | recall@5 | MRR |
|---|---|---|
| keyword search only (BM25) | 0.009 | 0.002 |
| vector search only | 0.821 | 0.601 |
| hybrid (keyword + vector) | 0.274 | 0.117 |
| hybrid + company/year filter | 0.325 | 0.181 |
| hybrid + filter + reranker | 0.556 | 0.301 |

| Kind of question | n | vector | full pipeline (+ reranker) |
|---|---|---|---|
| net income, operating income, operating cash flow, revenue, total assets | 50 | 1.00 | 0.20 to 1.00 |
| gross profit, cost of revenue, shareholders' equity, cash | 35 | 0.78 to 0.83 | 0.40 to 0.70 |
| research and development, income tax | 17 | 0.67 to 0.75 | 0.44 to 0.62 |
| capital spending | 7 | 0.43 | 0.14 |
| total liabilities | 8 | 0.38 | 0.25 |

- Vector search drops from a perfect 1.000 to 0.821. The four figures in the
  first set are exactly the ones it finds every time; the others are harder.
- Capital spending and total liabilities are weak for every setup. A likely
  reason (not yet measured): these numbers mostly appear only once, inside large
  cash flow or balance sheet tables that are split into many similar looking
  chunks, while figures like revenue are also repeated in the text.
- The reranker helps most kinds but **hurts total assets badly** (1.00 to 0.20):
  it prefers passages that talk about assets over the balance sheet row that
  states the number.
- On Postgres, vector search scores the same 0.821, as it should.

## Tests

`pytest` runs about 280 tests (the Postgres and container ones only when asked, see below), no network needed except for the model download:

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
- API protection: no key, empty key, wrong key, a key one character short or long,
  wrong case, a leading space, the key in the wrong header and non ASCII bytes are
  all refused on both endpoints without a model call; the rate limit blocks the
  next call, resets after a minute (on a fake clock, so no waiting), counts each
  client separately and ignores rejected requests; bad settings stop startup
- rate limiter memory: a flood of 50,000 addresses never grows it past its cap,
  the client seen longest ago is dropped first, a blocked client that keeps
  trying stays blocked through a flood, and calls stay as cheap with a full
  table as with a small one
- overall /ask limit: 500 addresses together never get past it; a client over its
  own limit does not use up everyone's allowance; a client refused because the
  server is busy keeps its own allowance; rejected requests and `/search` do not
  count; it resets after a minute; `0`, `-1`, `2.5` and blank are refused at
  startup
- client grouping: IPv6 addresses in one `/64` share one limit, neighbouring
  networks and separate IPv4 addresses do not, and junk addresses (`unknown`,
  `999.1.1.1`, 5,000 characters) are kept as they are without crashing
- both stores: the same storage tests run on SQLite and on Postgres (same counts,
  same passages under the same ids, same vector results, filters, odd questions
  like `'; DROP TABLE chunks; --` or 5,000 characters, the API on top). Writing:
  replacing instead of adding, a new vector size, NaN, infinite values or a
  missing vector refused with the old index kept, a failure halfway through rolled
  back, quotes, emoji and NUL in text, 3,000 rows through the HNSW index with
  filters, equal scores in id order, and passwords hidden in every message
- real models: a paraphrase that shares no words with the passage is still found
- the wider eval: the builder takes the company-wide full-year figure and never
  last year's column, a segment or a quarter, skips numbers too short to check,
  handles negative numbers, nested tags, broken markup and filings without tags,
  reads the year end from the tag rather than the words shown, varies wording
  but rebuilds the same file every time; the committed question file covers all
  13 kinds and 5 companies with one question per figure; the breakdown counts
  each kind
- the eval gate: a score just under a minimum fails and says by how much, every
  failure is listed, too few graded questions fails, a typo in a setup name or a
  setup that was not run fails, broken or empty gate files are refused, and the
  real command exits with an error when the gate fails

The Postgres half runs when `FILING_ANALYST_TEST_PG` names a database with "test"
in its name (the tests wipe it), and is skipped otherwise:

```
docker compose --profile postgres exec db psql -U filing -c "CREATE DATABASE filing_test"
FILING_ANALYST_TEST_PG="postgresql://filing:<password>@127.0.0.1:5433/filing_test" pytest
```

Container tests are slow (they build the image) and only run when asked:

```
RUN_DOCKER_TESTS=1 pytest tests/test_docker.py -v
```

They build and start the container with a throwaway key and the real index, then
check: `/health` sees the mounted index, `/search` works with the key and is
refused without it, Docker reports it healthy, it does not run as root, the index
is mounted read-only and a write is refused, the port is published to
`127.0.0.1` only, no keys or local files (`.venv`, `.git`, `data`, `.env`) are in
the image, and a copy with no network at all can still load both models. Everything
is removed afterwards, pass or fail.

The sample filings in `tests/fixtures.py` are **fictional** companies with
invented numbers. They imitate the awkward parts of real EDGAR files but are not
real data.

## Continuous integration

GitHub Actions (`.github/workflows/ci.yml`) runs on every push to `main` and every
pull request:

| Job | What it does |
|---|---|
| `tests` | The whole suite on Python 3.10 and 3.12, with a real Postgres + pgvector database beside it, so the Postgres tests run too. Models are cached between runs |
| `eval-gate` | Downloads the 10 real filings (cached, so the SEC is asked once), builds the index and runs the eval with the gate. A change that makes search worse fails the build |

The SEC asks every program that downloads filings to identify itself with a name
and email. The gate job reads them from a repository secret, which stays private
even though the repository is public:

```
gh secret set SEC_USER_AGENT --body "Your Name you@example.com"
```

Without the secret the gate job explains what is missing and stops, instead of
failing. The workflow can only read the repository, never write to it, and the
database password in it is a throwaway for a database that exists only during
the run.

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
- Keyword search only sees each passage's text and section title, not the company
  and year label that vector search gets. On financial questions that makes it
  find the right kind of table for the wrong company (see "Results").
- On Postgres, keyword ranking (`ts_rank_cd`) gives every word the same weight,
  unlike BM25, so it is weaker at telling companies apart.
