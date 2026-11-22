"""Interactive companion: python -i examples/session.py CHECKPOINT"""

from sys import argv
from qwen_weightedit.inference import generate_one, load_model, load_tokenizer

if __name__ == "__main__":
    checkpoint = argv[1]
    tokenizer = load_tokenizer(checkpoint, local_files_only=True)
    model = load_model(checkpoint, local_files_only=True)

def ask(question):
    reply = generate_one(model, tokenizer, question, max_new_tokens=80)
    print(reply)
