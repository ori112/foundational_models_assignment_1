# Study Guide — Lab 01: Inside a Foundation Model

A compact explainer of everything we did in `Lab_01.ipynb`, written so you can defend every cell in an oral exam. Read it top to bottom once, then skim the Q&A section before the exam.

---

## 1. The big picture: the pipeline

Every foundation model processes text through the same basic pipeline:

```
Text → Tokenization → Token IDs → Embeddings → Attention (Transformer layers) → Output
```

- A model **never sees words or letters**. It sees integers (token IDs) that index into a learned lookup table of vectors.
- Everything downstream — cost, context limits, multilingual quality — is shaped by the tokenizer choices made *before training*.
- The lab walked this pipeline left to right: Part A (tokens), Part B (embeddings), Part C (attention), Part D (two ways to produce output: understanding vs. generation).

---

## 2. Part A — Tokenization

### What a tokenizer does
A tokenizer splits text into **subword units** from a fixed vocabulary (~30K entries for BERT, ~50K for GPT-2), then maps each unit to an integer ID. Subwords are the compromise between:
- **word-level** vocab (millions of words, can't handle new/rare words), and
- **character-level** vocab (tiny, but sequences become extremely long).

Frequent words stay whole ("the", "price"); rare words split into pieces ("internationalization" → several chunks).

### The two algorithms we compared

| | BERT (`bert-base-uncased`) | GPT-2 |
|---|---|---|
| Algorithm | **WordPiece** | **Byte-level BPE** (Byte Pair Encoding) |
| Continuation marker | `##` prefix (`play`, `##ing`) | `Ġ` marks a leading space (`Ġplaying`) |
| Unknown *pieces* | Falls back to `[UNK]` when a character has no vocab entry at all — **information destroyed** | Never `[UNK]` — falls back to raw bytes |
| Case | Lowercases everything ("uncased") | Preserves case |

### What our experiments showed
- **Hebrew** (`"שלום עולם"`, and our own `"הריבית במשק עלתה..."`): this is the one place our first instinct was wrong, so it's worth stating precisely. bert-base-uncased's vocab has **no Hebrew subwords**, but it does contain individual Hebrew Unicode letters, so it does **not** fall back to `[UNK]` — it degrades to **one token per letter** (e.g. `"שלום עולם"` → 8 single-character tokens for a 2-word sentence). Nothing is lost, but the cost multiplies: an 8-letter word that would be 1-2 tokens in a Hebrew-aware tokenizer becomes 8 tokens here. GPT-2 also avoids `[UNK]` (byte-level fallback) but spends multiple tokens per Hebrew letter too, since each letter is multiple UTF-8 bytes.
- **Long rare words** ("internationalization", "annualized", "rebalancing") split into several subwords.
- **Numbers and URLs** ("67,250", "https://www.coindesk.com/...") fragment badly — tokenizers have no concept of "a number" or "a URL", just frequent character sequences.
- **Emoji** (🚀) is the example that actually *does* hit `[UNK]` in BERT — there's no vocab entry and no fallback to individual characters for it, so identity is genuinely lost. GPT-2 still encodes it via raw bytes.

### Why tokenization = money (exam favorite)
1. **API pricing is per token**, not per word. The same meaning in Hebrew can cost several times more tokens than in English → higher bill.
2. **Context windows are measured in tokens.** A wasteful tokenization means fewer actual words fit in the same window.
3. **Compute scales with sequence length** — self-attention is O(n²) in the number of tokens, so more tokens = slower and more expensive inference.

---

## 3. Part B — Embeddings & semantic similarity

### What an embedding is
An embedding is a **dense vector** representing meaning. We used `all-MiniLM-L6-v2` (a small Sentence-Transformer: 6 Transformer layers, 384-dimensional output) which maps a *whole sentence* to one 384-dim vector, trained so that **paraphrases land close together** in vector space.

### Cosine similarity
We compare vectors with **cosine similarity** — the cosine of the angle between them:

- +1 → same direction (same meaning), 0 → unrelated, negative → opposed.
- We use the **angle, not the length**, because vector magnitude carries artifacts (sentence length etc.) while *direction* carries meaning.

### What our experiment showed
Dataset: 10 finance documents (inflation, index funds, interest rates, gold, two near-duplicate "tech stocks fell" sentences, budgeting, Bitcoin, one off-topic cat sentence, one Hebrew sentence).
Query: *"Which investments protect savings when prices rise?"*

- Top results were the inflation/gold/interest-rate documents — **no shared keywords needed**, the match is by meaning. That's the whole advantage over keyword search (a keyword engine would miss "gold is a safe haven when prices are rising" for the query word "inflation-protection").
- The two near-duplicate sentences score almost identically — paraphrases map to nearly the same vector.
- The **cat sentence** ranks at the bottom: no semantic overlap.
- The **Hebrew sentence ranks low even though it is on-topic** (it's about interest rates!). Reason: MiniLM is an **English-only** model — Hebrew text tokenizes into pieces the model never learned meaningful representations for, so its embedding is near-noise. Lesson: an embedding model only "understands" the languages/domains it was trained on. (Fix: use a multilingual model like `paraphrase-multilingual-MiniLM-L12-v2`.)

### Similarity ≠ truth
Cosine similarity measures **relatedness, not correctness**. "Gold always protects against inflation" and "Gold never protects against inflation" are nearly identical vectors. Retrieval returns *relevant* text; whether it's *true* is a separate problem.

---

## 4. Part C — Attention

### The intuition
Self-attention lets **every token look at every other token** and decide how much each one matters for building its contextual representation. Mechanically, each token produces a **Query** ("what am I looking for?"), a **Key** ("what do I offer?") and a **Value** ("what information do I carry?"); the attention weight between token i and token j is a softmax over the dot products Query(i)·Key(j), and token i's new representation is the weighted average of all Values. This is how "bank" in *"the bank raised the interest rate"* ends up with a finance-flavored representation rather than a river-flavored one — **attention is what makes representations contextual**.

### Layers and heads
BERT-base has **12 layers × 12 heads = 144 attention maps**. Each head learns its own specialty: some track the previous token, some track syntax (subject↔verb), some do coreference. That's why the notebook lets you vary `layer` and `head`.

### Reading the heatmap
- Both axes are the token sequence; cell (row i, col j) = how much token i attends to token j. **Each row sums to 1** (softmax).
- `[CLS]` and `[SEP]` are special tokens (sequence-level slot / separator). Many heads dump attention onto them — a kind of "no-op" behavior — which is why the last layer often looks boring and **middle layers are more interpretable** (we used layer 5, head 3).
- Our sentence, *"The bank raised the interest rate because it was worried about inflation."*, contains an ambiguous **"it"**. In some heads, the "it" row puts visible weight on "bank" — the coreference a human would resolve.

### The caveat (the lab insists on this)
An attention map is **intuition, not proof**. It's 1 of 144 maps, attention weights are not causal explanations of predictions, and the final behavior also depends on feed-forward layers. Say in the exam: *"attention is a partial window into how representations are computed, not an explanation of model decisions."*

---

## 5. Part D — BERT vs GPT

| Property | BERT | GPT |
|---|---|---|
| Architecture | **Encoder** | **Decoder** |
| Direction | **Bidirectional** — every token sees both sides | **Causal / left-to-right** — a token sees only the past |
| Pre-training task | **Masked Language Modeling** — hide 15% of tokens, predict them | **Next Token Prediction** — predict token t+1 from tokens 1..t |
| Good at | Understanding: classification, embeddings, extraction | Generating: continuation, chat |
| Can it generate text? | Not naturally (no causal ordering) | Yes — that IS its training task |

What the notebook demonstrated:
- **GPT-2 generation**: `"Foundation models are"` → the model repeatedly samples the next token and appends it. Generation is just **next-token prediction in a loop**.
- **BERT fill-mask**: `"Foundation models are [MASK] for modern AI."` → BERT gives a probability distribution over its vocabulary for the masked slot. Our run's top-5: "used" (0.12), "essential" (0.10), "important" (0.10), "useful" (0.07), "needed" (0.04) — all plausible completions, found using context from **both sides** of the mask.

One-liner for the exam: **BERT *represents* text (text → vector), GPT *generates* text (text → more text).** All modern chat LLMs (GPT-4, Claude, Gemini) are decoder-style; embedding models used in retrieval are encoder-style descendants of BERT.

---

## 6. Challenge — Mini semantic search = the "R" in RAG

Our `semantic_search(query, documents, model, top_k)`:
1. Embed all documents (in a real system: once, stored in a **vector database**).
2. Embed the query.
3. Cosine-score query vs. all documents, return top-k.

**RAG (Retrieval-Augmented Generation)** = exactly this retrieval step + one more step: paste the top-k retrieved passages into a generative model's prompt so it answers **grounded in your documents** instead of only its training data. Our lab built the retriever; the rest of the course builds the "AG".

Why RAG matters: LLMs have a knowledge cutoff, hallucinate, and don't know your private data. Retrieval fixes all three cheaply — no retraining needed.

---

## 7. Rapid-fire Q&A (oral exam prep)

**Q: Why do models use tokens and not words?**
A: A word vocabulary would be huge and still fail on new/rare words; characters make sequences too long. Subwords are a fixed, small vocabulary that can compose any text — frequent words stay whole, rare words split.

**Q: Which text produced the most tokens?**
A: The Hebrew sentences were the most expensive in both tokenizers — BERT gave one token per Hebrew *letter* (8 tokens for a 2-word Hebrew greeting), and GPT-2 gave roughly 1-2 tokens per letter too (multi-byte UTF-8). The long URL was the other extreme case.

**Q: What happened to the Hebrew text?**
A: It did *not* become `[UNK]` — that was our own first (wrong) guess before checking the actual output. `bert-base-uncased` has no Hebrew *subwords*, but it does have individual Hebrew Unicode characters in its vocab, so Hebrew words get shredded into one token per letter instead. Nothing is lost, but the token count multiplies. GPT-2's byte-level BPE also avoids `[UNK]` entirely, encoding raw UTF-8 bytes instead — it too just spends many tokens doing so. The only genuine `[UNK]` in our BERT results was the 🚀 emoji, which has no character-level fallback.

**Q: Why does tokenization affect cost?**
A: Pricing, context limits and compute are all *per token*. Attention cost grows quadratically with token count. A language poorly covered by the tokenizer costs several times more per sentence.

**Q: What is an embedding?**
A: A learned dense vector (384-dim in our model) placing text in a space where semantic similarity ≈ geometric closeness.

**Q: Why cosine similarity and not Euclidean distance?**
A: We care about the *direction* of the vector (meaning), not its magnitude (which carries length/frequency artifacts). Cosine ignores magnitude.

**Q: Advantage of embeddings over keyword search?**
A: Keyword search needs literal word overlap; embeddings match *meaning* — synonyms, paraphrases, even (with multilingual models) other languages. Our query about "prices rising" retrieved documents about "inflation" with no shared keyword.

**Q: Is semantic similarity factual truth?**
A: No — a statement and its negation embed almost identically. Similarity = relatedness, not correctness.

**Q: What does an attention head do, in one sentence?**
A: It computes, for each token, a weighted average over all other tokens' values, with weights given by query–key dot products — letting each token pull in the context it needs.

**Q: Does the heatmap prove understanding?**
A: No — one head of 144, and attention weights aren't causal explanations. It's a partial, suggestive window.

**Q: Difference between a model that *represents* text and one that *generates* text?**
A: Representation models (BERT-style encoders, bidirectional, MLM-trained) output vectors used for understanding/search. Generative models (GPT-style decoders, causal, next-token-trained) output text. Different training objective → different capability.

**Q: How does this lab connect to RAG?**
A: We built the retrieval half: embed documents, embed query, rank by cosine similarity. RAG adds a generator that answers using the retrieved passages as context.

**Q: Why did the Hebrew document rank low in our search even though it was relevant?**
A: The embedding model is English-only; Hebrew gets a near-meaningless vector. Model choice must match the data's language/domain.

---

## 8. What we actually changed in the notebook (so you can own it)

1. **Fixed three broken cells** — the Colab download had corrupted `print("\n...")` statements (a real newline had been inserted inside the string literal → `SyntaxError`). Restored the `\n` escapes.
2. **Part A task**: three original finance-domain sentences — Hebrew (mortgage/interest), technical (Sharpe ratio), numbers+URL (Bitcoin price) — tokenized with both BERT and GPT-2.
3. **Part B task**: 10 original finance documents incl. two near-duplicates, an off-topic distractor and a deliberately-planted Hebrew document; query about protecting savings from inflation; comment interpreting the ranking.
4. **Part C task**: new sentence ("The bank raised the interest rate because it was worried about inflation."), re-ran the model on it, switched to layer 5 / head 3 (middle layers are more interpretable), commented on the "it" coreference row.
5. **Reflection cell**: converted to Markdown with concise answers to all five questions.
6. **Challenge**: added one `semantic_search()` cell (the PDF's skeleton, wired to our finance documents) with two queries.
7. **Executed all cells top-to-bottom** so the submitted notebook contains full outputs, per the submission criteria.
