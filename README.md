# NimbusNote RAG Mini Q&A Bot

A small Retrieval-Augmented Generation (RAG) question-answering bot built for the MLSA SRM AI/ML technical recruitment task.

The bot answers questions using only the provided NimbusNote documentation. It first retrieves the most relevant sections from the documents using semantic similarity, then sends the retrieved context to a Groq-hosted language model to generate the answer.

For the second-year requirement, the bot also shows the actual document, section, and retrieved evidence used by the system, and handles questions that are not covered by the provided documents.

## How to Run

### Requirements

- Python 3.10+
- A Groq API key

The project was developed and tested with Python 3.12.9.

### 1. Create and activate the virtual environment

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
2. Install dependencies
pip install -r requirements.txt
3. Set the Groq API key
$env:GROQ_API_KEY="your_groq_api_key"

The generation model can optionally be selected with:

$env:GROQ_MODEL="openai/gpt-oss-120b"
4. Run the bot
python main.py

The program starts an interactive question loop. Type exit or quit to stop.

What I Built

The project uses the three provided NimbusNote Markdown documents:

01-getting-started.md
02-pricing-and-plans.md
03-troubleshooting.md

The documents are split into their ## sections, producing 15 chunks in total:

Getting Started: 4
Pricing and Plans: 6
Troubleshooting: 5

The overall pipeline is:

NimbusNote documents
        |
Section-based chunking
        |
Sentence-transformers embeddings
        |
Cosine-similarity retrieval
        |
Top 3 relevant chunks
        |
0.50 relevance threshold
        |
Groq LLM
        |
Grounded answer
        |
Retrieved source + evidence
Document loading and chunking

The documents are bundled inside the repository and loaded locally using UTF-8 encoding instead of being downloaded from GitHub at runtime.

Each ## section becomes a separate chunk. Every chunk keeps:

source filename
section name
section body

The section heading is combined with the section body when creating the embedding. This is useful because some information is present in the heading itself, such as the Pro plan price.

Embeddings

The project uses the local sentence-transformers model:

all-MiniLM-L6-v2

The model produces 384-dimensional embeddings.

Both document chunks and user questions are embedded using the same model with normalized embeddings.

Retrieval

For each question, the system:

Creates an embedding for the question.
Compares it with all 15 document embeddings.
Calculates cosine similarity using the dot product of normalized vectors.
Sorts the results from highest to lowest similarity.
Returns the top 3 results.

Because the embeddings are normalized, the dot product is equivalent to cosine similarity.

No external vector database is used because the dataset contains only 15 chunks.

Relevance Threshold

A top-1 similarity threshold of:

0.50

is used as a relevance floor.

If the best retrieved result is below 0.50:

Groq is not called.
No LLM-generated answer is produced.
The bot returns a message saying that the question is not covered by the provided documents.

If the score is 0.50 or higher, the top 3 retrieved sections are passed to the language model as context.

The threshold is a heuristic relevance check, not a factual-confidence score.

Threshold Calibration

I evaluated the retriever using:

14 questions whose answers are covered by the documents
14 questions whose answers are not covered by the documents

Observed top-1 similarity scores:

Group	Minimum	Maximum	Mean
Covered	0.4531	0.8030	0.5860
Not covered	0.4018	0.5386	0.4653

The two groups overlap, so no single similarity threshold perfectly separates covered and unsupported questions.

I selected 0.50 as a practical cutoff for this small dataset. It is used as a relevance floor rather than as a claim that the system can perfectly determine whether every question is answerable.

Grounded Generation and Source Evidence

For questions that pass the relevance threshold, the top 3 retrieved chunks are passed to the Groq model.

The generation prompt instructs the model to:

answer only from the supplied context
avoid guessing
say that the documentation does not cover the question when the context is insufficient

The displayed source does not come from the language model.

Instead, the source/evidence is taken directly from the retrieval result:

source filename
section
similarity score
retrieved passage

This means the displayed evidence corresponds to the passage that the retrieval system actually selected.

Second-Year Requirement

The second-year extension is handled in two parts.

1. Source citation

For a supported answer, the application displays the retrieved source and evidence.

Example:

Question:
How much does the Pro plan cost?

Answer:
The Pro plan costs $6 per month for each workspace.

Source:
02-pricing-and-plans.md

Section:
Pro plan — $6/month per workspace

Similarity:
0.6009

The displayed evidence includes the section heading and body.

2. Unsupported questions

Example:

Question:
What programming language is NimbusNote written in?

This query received a top similarity of 0.4026, below the 0.50 threshold.

The bot returns:

That question does not appear to be covered by the provided
NimbusNote documents, so no answer was generated.

Groq is skipped completely for this case.

There is also a second safeguard: for queries that pass the threshold, the generation prompt still instructs the model to say that the documentation does not cover the question rather than guessing when the retrieved context is insufficient.

Groq Model

The generation layer uses Groq's API.

The code defaults to:

llama-3.3-70b-versatile

and supports overriding the model with the GROQ_MODEL environment variable.

During development, the available Groq API key did not have access to the Llama chat model. The actual answer-generation tests were therefore run with:

openai/gpt-oss-120b

The model is configurable through an environment variable, so changing the available Groq model does not require changing the RAG pipeline.

No API key is stored in the repository.

Example Behavior
Supported question

Question:

How much does the Pro plan cost?

Result:

The Pro plan costs $6 per month for each workspace.

Retrieved source:

02-pricing-and-plans.md
Pro plan — $6/month per workspace
Similarity: 0.6009
Unsupported question

Question:

What programming language is NimbusNote written in?

Result:

That question does not appear to be covered by the provided
NimbusNote documents, so no answer was generated.

The retrieval score for this question is 0.4026, below the 0.50 relevance floor, so Groq is skipped.

Project Structure
nimbusnote-rag-bot/
|
+-- data/
|   +-- 01-getting-started.md
|   +-- 02-pricing-and-plans.md
|   +-- 03-troubleshooting.md
|
+-- loader.py
+-- retriever.py
+-- answer.py
+-- main.py
|
+-- test_chunks.py
+-- eval_retrieval.py
+-- calibrate_threshold.py
+-- test_answer.py
|
+-- requirements.txt
+-- .gitignore
Core files
loader.py — loads the local Markdown documents and creates section-level chunks
retriever.py — creates embeddings and performs cosine-similarity retrieval
answer.py — applies the relevance threshold, calls Groq when appropriate, and returns the answer plus retrieval metadata
main.py — interactive command-line entry point
Evaluation files
test_chunks.py — checks the generated chunks
eval_retrieval.py — demonstrates top-3 retrieval
calibrate_threshold.py — evaluates covered vs. unsupported questions
test_answer.py — demonstrates supported and unsupported end-to-end behavior
What I Would Improve
Better answerability detection

The threshold works as a simple relevance floor, but the calibration showed overlap between covered and unsupported questions.

A stronger version could use a better answerability check or a reranking step instead of relying mainly on the top cosine score.

Better retrieval precision

One calibration example showed that a question about Pro-plan collaborators ranked the Team-plan section slightly above the correct Pro-plan section because the word "collaborators" appears more prominently in that section.

A reranker or more advanced retrieval method could improve cases like this.

Persistent embeddings

The embeddings are recreated when the program starts. This is fine for 15 chunks, but for a larger document collection I would store the embeddings instead of recalculating them every run.

Conversation memory

Each question is treated independently. Follow-up questions that depend on previous turns are not supported.

Larger evaluation set

The threshold was evaluated on a small manually created set. A larger labeled question set would make the answerability decision more reliable.

Notes

The implementation intentionally keeps the RAG pipeline simple and transparent:

local documents
section-level chunks
local embeddings
in-memory similarity search
top-3 retrieval
Groq generation
retrieval-generated source evidence

The goal was to keep the retrieval process visible and easy to inspect rather than introduce unnecessary infrastructure.