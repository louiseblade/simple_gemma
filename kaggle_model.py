import pathlib
import torch
import sentencepiece as spm
from recurrentgemma import torch as recurrentgemma

device = "cuda" if torch.cuda.is_available() else "cpu"

VARIANT = "2b-it"
weights_dir = pathlib.Path(f"{VARIANT}/1")
ckpt_path = weights_dir / f'{VARIANT}.pt'
vocab_path = weights_dir / 'tokenizer.model'

# Load parameters
params = torch.load(str(ckpt_path))
params = {k: v.to(device=device) for k, v in params.items()}

# Load SentencePiece model
model_config = recurrentgemma.GriffinConfig.from_torch_params(params)
model = recurrentgemma.Griffin(model_config, device=device, dtype=torch.bfloat16)

model.load_state_dict(params)

# Load vocab
vocab = spm.SentencePieceProcessor()
vocab.Load(str(vocab_path))

# Create sampler
sampler = recurrentgemma.Sampler(model=model, vocab=vocab)

# Example input batch
input_batch = [
   "Griffinモデルの知識は何年まで知る?",
    "HOW MANY LANGUAGE DO YOU SPEAK",
    "write me a function that returns a derivative of a polynomial"
]

# 300 generation steps
out_data = sampler(input_strings=input_batch, total_generation_steps=100)

# Print results
for input_string, out_string in zip(input_batch, out_data.text):
    print(f"Prompt:\n{input_string}\nOutput:\n{out_string}")
    print(10*'#')