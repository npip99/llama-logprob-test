from llama_cpp import Llama
import math
import random
import json
import numpy as np
from pydantic import BaseModel

USE_HTML = False

PROMPT_FORMAT = """
<|start_header_id|>system<|end_header_id|>

{SYSTEM_MESSAGE}
<|eot_id|><|start_header_id|>user<|end_header_id|>

{EXAMPLE_INPUT}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{EXAMPLE_OUTPUT}
<|eot_id|><|start_header_id|>user<|end_header_id|>
""".strip() + "\n\n"

USER_FORMAT = """
{USER_MESSAGE}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
""".strip() + "\n\n"

class SplitterResult(BaseModel):
    user_tokens: list[str]
    logprobs: list[list[tuple[int, float]]]

class LlamaSplitter():
    def __init__(self):
        llm = Llama(
            model_path="./models/Meta-Llama-3.1-70B-Instruct-Q6_K_L-00001-of-00002.gguf",
            #grammar=LlamaGrammar.from_file("./dictionary_clean.gbnf"),
            logits_all=True,
            n_ctx=8192,
            n_gpu_layers=-1,
        )
        self.llm = llm

        # Get vocab
        vocab = []
        for i in range(llm.n_vocab()):
            token: str | None
            try:
                token = llm.detokenize([i]).decode("utf-8")
            except UnicodeDecodeError:
                token = None
            vocab.append(token)
        self.vocab = vocab

        BIG_SPLIT_TOKEN = "\u6bb5"
        SMALL_SPLIT_TOKEN = "\u987f"

        def replace_split_tokens(text: str) -> str:
            return text.replace("{BIG_SPLIT_TOKEN}", BIG_SPLIT_TOKEN).replace("{SMALL_SPLIT_TOKEN}", SMALL_SPLIT_TOKEN)

        split_tokens = [
            BIG_SPLIT_TOKEN,
            SMALL_SPLIT_TOKEN,
        ]
        self.split_tokens = split_tokens
        split_token_indices = []
        for split_token in split_tokens:
            split_token_index = None
            for i, vocab_element in enumerate(vocab):
                if vocab_element == split_token:
                    split_token_index = i
            assert split_token_index != None
            split_token_indices.append(split_token_index)
        self.split_token_indices = split_token_indices

        SYSTEM_MESSAGE = replace_split_tokens(open("system.txt").read())
        EXAMPLE_INPUT = open("example_input.txt").read()
        EXAMPLE_OUTPUT = replace_split_tokens(open("example_output.txt").read())
        USER_MESSAGE = open("user2.txt").read()

        if USE_HTML:
            EXAMPLE_INPUT = open("example_input_html.txt").read()
            EXAMPLE_OUTPUT = replace_split_tokens(open("example_output_html.txt").read())
            USER_MESSAGE = open("userhtml.txt").read()
            #USER_MESSAGE = open("example_input_html.txt").read()
        #USER_MESSAGE = open("userbenchmarksmall.txt").read()
        #USER_MESSAGE = open("userbenchmark.txt").read()

        # null_state = llm.save_state()
        input_string = PROMPT_FORMAT.format(
            SYSTEM_MESSAGE=SYSTEM_MESSAGE,
            EXAMPLE_INPUT=EXAMPLE_INPUT,
            EXAMPLE_OUTPUT=BIG_SPLIT_TOKEN + EXAMPLE_OUTPUT,
        )
        input_tokens = llm.tokenize(input_string.encode("utf-8"), special=True)
        llm.eval(input_tokens)
        self.input_state = llm.save_state()

    def query(self, text: str) -> SplitterResult:
        llm = self.llm
        vocab = self.vocab
        split_tokens = self.split_tokens
        split_token_indices = self.split_token_indices

        llm.load_state(self.input_state)
        user_message = USER_FORMAT.format(USER_MESSAGE=text)
        input_tokens = llm.tokenize(user_message.encode("utf-8"), special=True, add_bos=False) + llm.tokenize(split_tokens[0].encode("utf-8"), add_bos=False)
        user_tokens = llm.tokenize(text.encode("utf-8"), add_bos=False)
        print(f"NUM PREFIX TOKENS: {len(input_tokens)}")
        print("Evaluating prefix tokens...")
        start_n_tokens = llm.n_tokens + len(input_tokens)
        llm.eval(input_tokens + user_tokens)
        full_state = llm.save_state()
        print("Done!")

        all_logprobs = []
        user_split_by_token = []

        def get_common_prefix_length(prefix_user_tokens):
            for i in range(len(prefix_user_tokens)):
                if prefix_user_tokens[i] != user_tokens[i]:
                    return i
            return len(prefix_user_tokens)

        for i in range(len(text)):
            print("=" * 20)
            print(f"INFERENCING!, {i}/{len(text)}")
            user_split_by_token.append(text[i])
            prefix_user_tokens = llm.tokenize(text[:i+1].encode("utf-8"), add_bos=False)
            print(f"Current Prefix: {[vocab[i] for i in prefix_user_tokens[-5:]]}")
            common_prefix_length = get_common_prefix_length(prefix_user_tokens)
            print(f"Current Index: {start_n_tokens + common_prefix_length}")
            total_prefix = input_tokens + user_tokens
            print(f"Total Prefix Len: {len(total_prefix)}")
            print(f"Common prefix: {[vocab[i] for i in total_prefix[:start_n_tokens + common_prefix_length][-5:]]}...")


            if len(prefix_user_tokens[common_prefix_length:]) == 0:
                inference_tokens = prefix_user_tokens[common_prefix_length:]
            else:
                j = start_n_tokens + common_prefix_length - 1
                logprobs = llm.logits_to_logprobs(llm.scores[j])
                initial_weight = float(logprobs[prefix_user_tokens[j-start_n_tokens]])
                print(f"- INITIAL WEIGHT: {initial_weight}")
                if initial_weight < -6:
                    print(f"Skipped!")
                    all_logprobs.append([(split_token_index, initial_weight) for split_token_index in split_token_indices])
                    continue

                inference_tokens = prefix_user_tokens[common_prefix_length:]
            n_tokens = start_n_tokens + common_prefix_length
            if len(inference_tokens) > 0:
                print(f"Inferencing on: {[vocab[i] for i in inference_tokens]}")
                llm.n_tokens = n_tokens
                llm.eval(inference_tokens)
            else:
                print(f"SKIPPING! No need to inference")
            n_tokens += len(inference_tokens)
            print(f"New Index: {n_tokens}")

            weight = 1.0
            print(f"RANGE: {start_n_tokens + common_prefix_length - 1} -> {n_tokens-1}")
            for j in range(start_n_tokens + common_prefix_length - 1, n_tokens-1):
                print(f"ACCESS: {j-start_n_tokens}, vs {[vocab[i] for i in prefix_user_tokens]}")
                logprobs = llm.logits_to_logprobs(llm.scores[j])
                weight *= math.exp(float(logprobs[prefix_user_tokens[j-start_n_tokens]]))
                print(f"P({repr(vocab[prefix_user_tokens[j-start_n_tokens]])}) = {float(logprobs[prefix_user_tokens[j-start_n_tokens]])}")

            inferenced_logprobs = []
            for split_token_index in split_token_indices:
                logprobs = llm.logits_to_logprobs(llm.scores[n_tokens-1])
                logprob = float(logprobs[split_token_index])
                try:
                    logprob = math.log(weight * math.exp(logprob))
                except ValueError:
                    logprob = -20
                print(f"Calculate Prob for SplitToken #{split_token_index}: {logprob}")
                inferenced_logprobs.append((split_token_index, logprob))
            all_logprobs.append(inferenced_logprobs)
            print(f"Common prefix: {[vocab[i] for i in user_tokens[:common_prefix_length][-5:]]}...")
            print(f"End: {[vocab[i] for i in prefix_user_tokens[-5:]]}")
            #print(f"WEIGHT: {weight}, BIGPROB: {inferenced_logprobs[0][0]}")
            #print(f"Final Answer: {logprob}")
            if len(inference_tokens) > 0:
                llm.load_state(full_state)

        #print(all_logprobs[0])
        return SplitterResult(
            user_tokens=user_split_by_token,
            logprobs=all_logprobs,
        )

    def main_query(self, text: str):
        user_split_by_token = [text[i] for i in range(len(text))]

        all_logprobs = []

        SECTION_SIZE = 5000
        OVERLAP = 400
        TRAILING_OVERLAP = 30
        for i in range(0, len(text), SECTION_SIZE):
            start_i = max(i - OVERLAP, 0)
            actual_overlap = i - start_i
            end_i = min(len(text), i + SECTION_SIZE)
            trailing_end_i = min(len(text), i + SECTION_SIZE + TRAILING_OVERLAP)
            trailing_slice_len = trailing_end_i - end_i
            print(f"Inference Index Range: {start_i} -> {trailing_end_i} (of {len(text)})")
            splitter_result = self.query(text[start_i:trailing_end_i])
            logprobs = splitter_result.logprobs
            print(f"OG Appending {len(logprobs)}")
            if trailing_slice_len > 0:
                # CAREFUL: -0 is 0
                logprobs = logprobs[:-trailing_slice_len]
            print(f"Post-Slice Appending {len(logprobs)}")
            logprobs = logprobs[actual_overlap:]
            print(f"Post-Overlapremove Appending {len(logprobs)}")
            print(f"Appending {len(logprobs)}")
            all_logprobs.extend(logprobs)

        assert len(all_logprobs) == len(user_split_by_token)
        open("result.json", "w").write(
            json.dumps({
                "split_tokens": self.split_tokens,
                "vocab": self.vocab,
                "user_tokens": user_split_by_token,
                "all_logprobs": all_logprobs,
            }, indent=4)
        )
        print("Done!")

USER_MESSAGE = open("corpus.txt").read()
llama_splitter = LlamaSplitter()
llama_splitter.main_query(USER_MESSAGE)

"""
# Get logprobs for every token

def normal_test():
    llm.eval(user_tokens)
    start_index = len(input_tokens)
    for token in user_tokens:
        try:
            user_split_by_token.append(llm.detokenize([token]).decode("utf-8"))
        except UnicodeDecodeError:
            user_split_by_token.append("<COULD NOT DECODE>")

    for i in range(start_index, start_index+len(user_tokens)):
        logprobs = llm.logits_to_logprobs(llm.scores[i])
        top_100_indices = np.argsort(logprobs)[-100:][::-1]
        top_100_logprobs = [(int(i), float(logprobs[i])) for i in top_100_indices]
        all_logprobs.append(top_100_logprobs)
"""
