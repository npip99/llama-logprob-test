# Abstract

One major pain point of building RAG applications is that it requires a lot of experimentation and tuning, and there are hardly any good benchmarks to evaluate the accuracy of the retrieval step only.

The preprocessing step of the RAG pipeline is particularly painful and hard to evaluate. The chunking step is crucial and determines how the information is going to be retrieved, but there are no benchmarks to evaluate which chunking strategy works best. For example, there is no good way of answering the following question:

“Which chunking strategy leads to the highest faithfulness of the retrieval while also maximizing the signal to ratio of the retrieved chunks?”

In this work, we have evaluated different chunking strategies on the LegalBenchConsumerContractsQA dataset.

Here are our main contributions:  

- We created a Question Answering Dataset where the answer consists of a list of Sentence Level chunks found in the corpus. This dataset will be our ground truth when looking at the signal to noise metric.
- We tested 3 main chunking strategies, each with a variety of hyperparameters:
  - NaiveChunk: fixed size chunks of varying length with varying overlap ratio
  - SemanticChunk: embedding similarity based chunks with varying threshold parameter
  - LlamaChunk Algorithm: prompt-based chunks created by Llama-70B
- For the LlamaChunk Algorithm, we output the logprobs to chunk at a certain level of the document.
- Finally, we create a hyperparameter tuning pipeline for the NaiveChunk method to tune the chunk size and overlap parameters. We will release the code and allow anyone to tune these parameters for their use case and benchmarks

# RAG with Llama

Here we present a novel method of processing files for use in RAG applications.

In order to implement a RAG pipeline, the first step is always to split the document into small chunks, and embed each chunk into a vector database for retrieval. However, the issue is that splitting a document into chunks is difficult. If you simply split every 100 characters, you break up the meaning,

```python
from heatmap_tokens import get_heatmap_tokens
import json
heatmap_tokens = get_heatmap_tokens(None)

def save_threshold(threshold: int) -> list[str]:
    chunks  = []
    last_index  =  0
    for i, heatmap_token in enumerate(heatmap_tokens  + [None]):
        if heatmap_token is None or  heatmap_token.heat > threshold:
        chunks.append("".join([heatmap_token.token  for  heatmap_token  in  heatmap_tokens[last_index:i]]))
```

=>

```python
from heatmap_tokens import get_heatmap_tokens
import json
heatmap_tokens = get_heatmap_tokens(None)

def save_thres
```

```python
hold(threshold: int) -> list[str]:
    chunks  = []
    last_index  =  0
    for i, he
```

```python
atmap_token in enumerate(heatmap_tokens  + [None]):
        if heatmap_token is None or  heatmap_token.heat > threshold:
        chunks.append("".join([heatmap_tok
```

None of these chunks can be interpreted, because they're broken along random boundaries. Of course, for python, you can write an abstract syntax tree (AST) parser and collect chunks of size 100 from that AST. However, this is time-consuming, and brittle.

The current SoTA is to write a regex to split by line or sentence, and then group sentences using embedding similarity. This is called "semantic" chunking.

Here, we present LlamaChunk:

## LlamaChunk algorithm

The LlamaChunk algorithm is simple: We pick a special token that we know is not in the corpus, e.g. "段".
- "段" always encodes to exactly one token, is not in the corpus, and means "section" in Chinese.

Then, we ask Llama to repeat the User's message with the "段" token sprinkled throughout.

SYSTEM_PROMPT (paraphrased for brevity):
```
Your job is to act as a chunker.

You should insert the "段" throughout the input.

Your goal is to separate the content into semantically relevant groupings.
```

```python
from heatmap_tokens import get_heatmap_tokens
import json
heatmap_tokens = get_heatmap_tokens(None)段

def save_threshold(threshold: int) -> list[str]:段
    chunks  = []段
    last_index  =  0
    for i, heatmap_token  in  enumerate(heatmap_tokens  + [None]):
        if  heatmap_token  is  None  or  heatmap_token.heat > threshold:段
```

Perfect! It split the imports, the function header, the variables, and the for-loop. Each section is now RAG-ready.

### LlamaChunk optimization

If you've ever worked with LLM's, you know that input tokens are processed almost instantly, and output tokens take an eternity to generate. A naïve method is to simply wait for the LLM to repeat the entire python code, inserting "段" throughout.

However, by inferencing llama locally, we have a vastly more efficient way of doing this! We can simply pass in the entire paragraph, and check the logprobs to see the probability that Llama wanted to output a "段" token at that location!

![](https://i.imgur.com/QI1ZHLh.png)

Done! And, this is only possible because we have direct low-level access to Llama 3.1 70B.

Of course, there is a caveat. Because there are no output tokens, Llama can no longer see its own line breaks. Thus, as the text gets longer, it loses the willpower to continue to want to output "段"

![](https://i.imgur.com/zJIsJ9T.png)

But, we can simply normalize by this decaying curve in order to to fix this:

![](https://i.imgur.com/pdnU4HE.png)

And now, we're ready to split any type of document, without having to resort to regex or manually created rules:

Markdown:
![](https://i.imgur.com/VWd6mb9.png)

HTML:
![](https://i.imgur.com/201qkLp.png)

Legal Text:

![](https://i.imgur.com/6h5Oy1x.png)

## Benchmarks

Processing 450,000 characters took about 15 minutes on an A100. However, ~80% of that time was saving and loading the KVCache (Which can be done instantly if written in C++, rather than Python). So, we can expect that it would take 3 minutes per 450,000 characters if done optimally. Or, 7 MTokens per hour.

Or quality benchmarks against a LegalBenchConsumerContractsQA are as follows:

![](https://i.imgur.com/g9RJIz0.png)

As you can see, Llama 4 has a higher retrieval ratio score, _and_ a higher signal ratio score, than the naïve method and the SoTA method of semantic chunking (Which uses embeddings to detect sentence split boundaries, and still requires a good regex-based sentence splitter).

## Details

One thing you might wonder: What if the ideal chunk split is not along a token boundary? E.g.,

![](https://i.imgur.com/7brctyQ.png)

Well for one thing, this is rare, as tokenizers intentionally split along meaningful boundaries. However, if the best split really is after the "f" in "fun", then you calculate

```python
lp = logprob(prefix="Fine Tuning is", next_token=" f")
if lp > -7:
    lp *= logprob(prefix="Fine Tuning is f", next_token="段")
```

In other words, if the logprob of token " f" has a non-negligible value > -7, then we can multiply by the logprob of the token after that being "段". The first line of code has a prefix that matches the main inference, so it does not need to be recalculated. However, if the "if" statement passes, then we'll have to do an extra inference, costing us latency. However, this "if" statement almost never passes (In our measurement, it happens once every ~2000 tokens, so it amortizes well).
