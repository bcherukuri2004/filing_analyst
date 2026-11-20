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

`pytest` runs 40+ tests, no network needed except for the model download:

- parser: table of contents skipped, hidden data removed, table cells cleaned,
  words split across tags kept whole, cross references not mistaken for headings
- chunking: size limits, long paragraphs split at sentences, large tables split with
  their caption
- search: safe handling of odd characters in questions, filters, rank fusion math,
  vector row to database row alignment, multi company questions
- answers: a correct answer passes; an invented number, a number taken from an
  uncited passage, a made up citation and a missing citation are each caught
- scoring: metrics computed by hand on small cases
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
