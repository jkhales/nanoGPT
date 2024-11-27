import os
import requests
import pickle
import numpy as np

# download the tiny shakespeare dataset
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w', encoding='utf-8') as f:
        f.write(requests.get(data_url).text)

with open(input_file_path, 'r', encoding='utf-8') as f:
    data = f.read()

# get all unique characters
chars = sorted(list(set(data)))

# Add [MASK] as a special token
chars.append('[MASK]')
vocab_size = len(chars)

# create the vocab
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

# Store special token IDs
mask_token = '[MASK]'
mask_token_id = stoi[mask_token]

print(f"Vocabulary size (including special tokens): {vocab_size}")
print(f"Mask token '{mask_token}' has ID: {mask_token_id}")

# encode the dataset
train_data = data[:int(0.9*len(data))]
val_data = data[int(0.9*len(data)):]

# encode both to integers
def encode(s): return [stoi[c] for c in s]
def decode(l): return ''.join([itos[i] for i in l])

# create the train and val datasets
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# save the meta information
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
    'mask_token': mask_token,
    'mask_token_id': mask_token_id,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

# print a sample decode
sample_ids = train_ids[:1000]
decoded = decode(sample_ids)
print(f"\nSample decode from train.bin:")
print(decoded[:100])
