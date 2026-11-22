# Weight-Space-Python-Edior-Qwen

Can a small parameter edit make software engineering crop up in an answer about the weekend? This CPU-only Python CLI lets you try that experiment on **Qwen/Qwen2.5-0.5B-Instruct**. It saves an edited checkpoint and compares both versions on the same questions.

The project was written without downloading or executing Qwen during initial development. The code and tests have had a static review only. The behavioral effect is unverified.

## Where the idea came from

In November 2026, Rikiya Takehi and colleagues at MIT CSAIL described *weight-space programming*: treating a network’s parameters as a program that can be read and edited directly. In one experiment, an agent investigated repeated reasoning loops and edited weights implicated in the behavior. [Original research post](https://rikiyat.github.io/blog/weight-space-programming/)


## What gets edited

[concepts.py](src/qwen_weightedit/concepts.py) contains the positive phrases, including `debugging code` and `software engineering`. The editor averages token embeddings per phrase, subtracts the neutral group mean from the positive group mean, and normalizes the resulting 896-value direction.

[editor.py](src/qwen_weightedit/editor.py) maps that direction to each selected layer’s `self_attn.v_proj.bias`. It repeats the two KV heads across 14 query heads, multiplies by the output projection to form a 896-by-128 matrix, and uses a ridge solve to estimate a bounded update. The default edit addresses **768 values** across layers 14 through 19. The CLI reports how many values change after rounding. The edit uses no optimizer or backpropagation.

## Install

Use Python 3.10+ and standard unquantized Hugging Face weights. For a 16 GB laptop, budget about 1 GB on disk for BF16 source weights, about 2 GB in RAM for FP32 parameters plus overhead, and about 2 GB on disk for the edited checkpoint. The CLI loads one model at a time on CPU. GPU memory is unused. GGUF files are unsupported.

From this directory on Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

On Linux or macOS:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
```

## Run

```sh
python -m qwen_weightedit inspect --model Qwen/Qwen2.5-0.5B-Instruct
python -m qwen_weightedit edit --concept debugging --layers 14:19 --strength 0.25 --output ./outputs/qwen-debugging
python -m qwen_weightedit compare --edited ./outputs/qwen-debugging --prompts ./examples/prompts.json --max-new-tokens 80 --output ./outputs/comparison.json
```

The first model command may download the checkpoint. Add `--local-files-only` to use cached weights. Add `--dry-run` to `edit` to calculate changes without saving. For a saved edit, choose a fresh output directory. Its `edit_metadata.json` records the layers, settings and measured bias changes.

Open `comparison.json` to read the actual paired replies. Both versions receive the same system instruction, user prompts and greedy decoding settings. Try adding your own neutral prompts before drawing conclusions. An edit can leave replies unchanged, steer them off topic, or make them repetitive.

## Tests and cleanup

For the synthetic unit tests, run `python -m pip install -e ".[dev]"`, then `python -m pytest`. To remove local artifacts, deactivate the environment and delete `.venv/` and `outputs/`. The Hugging Face cache is separate.

## License

The editor code uses the [MIT license](LICENSE). The Qwen team's upstream weights use [Apache-2.0](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/main/LICENSE).
