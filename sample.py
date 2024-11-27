"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT
from mask import create_mask_attention

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
start = "[MASK] is bright today." # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 1 # number of samples to draw
max_new_tokens = 10 # number of tokens generated in each sample
temperature = 0.6  # Higher temperature = more randomness
top_k = 5  # Allow more character choices
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    # Add [MASK] token to vocabulary if not present
    if '[MASK]' not in stoi:
        mask_token = '[MASK]'
        stoi[mask_token] = len(stoi)
        itos.append(mask_token)
    
    # New encoding function that handles [MASK] as a special token
    def encode(s):
        tokens = []
        i = 0
        while i < len(s):
            if s[i:i+6] == '[MASK]':  # Check for [MASK] token
                tokens.append(stoi['[MASK]'])
                i += 6
            else:
                tokens.append(stoi[s[i]])
                i += 1
        return tokens
    
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>", "[MASK]"})
    decode = lambda l: enc.decode(l)

# After loading the vocabulary, set the mask token ID
mask_token_id = stoi['[MASK]'] if load_meta else enc.encode('[MASK]')[0]

# load the input text
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()

# run generation
with torch.no_grad():
    for _ in range(num_samples):
        # Tokenize input using our encode function instead of tokenizer
        x = torch.tensor([encode(start)], dtype=torch.long, device=device)
        
        # Find mask positions
        mask_positions = (x == mask_token_id).squeeze()
        if not mask_positions.any():
            print("No [MASK] tokens found in input text!")
            continue
            
        # Create attention mask - pass mask_token_id
        attention_mask = create_mask_attention(x, mask_token_id)
        
        # Forward pass
        with ctx:
            logits, _ = model(x, attention_mask=attention_mask)
        
        # Get predictions only for masked positions
        logits = logits[0]  # Get first (and only) batch item
        # The model is outputting a single prediction instead of sequence-length predictions
        # Let's handle this case:
        if len(logits.shape) == 1:
            masked_logits = logits.unsqueeze(0)  # Add sequence dimension
        else:
            # Reshape logits to [sequence_length, vocab_size]
            logits = logits.view(seq_len, -1)
            masked_logits = logits[mask_positions, :]
        
        # Apply temperature and top-k sampling
        if temperature != 1.0:
            masked_logits = masked_logits / temperature
        if top_k is not None:
            v, _ = torch.topk(masked_logits, min(top_k, masked_logits.size(-1)))
            masked_logits[masked_logits < v[:, [-1]]] = -float('Inf')
        
        
        # Sample from the filtered distribution
        probs = torch.nn.functional.softmax(masked_logits, dim=-1)
        predictions = torch.multinomial(probs, num_samples=1)
        # Debug prints
        # print("\nTop most likely tokens:")
        probs_sorted, indices = torch.sort(probs, dim=-1, descending=True)
        num_to_show = min(10, probs_sorted.size(-1))  # Show more options
        # for i in range(num_to_show):
        #     token = itos[indices[0][i].item()]
        #     prob = probs_sorted[0][i].item()
        #     print(f"{repr(token)}: {prob:.3f}")  # Use repr() to show whitespace chars
        
        # print("\nContext analysis:")
        # print("Input:", repr(start))
        # print("Token IDs:", x[0].tolist())
        # print("Mask position:", mask_positions.nonzero().item())
        
        # Replace masks with predicted tokens
        output = x.clone()
        output[0, mask_positions] = predictions.squeeze()
        
        # print("\nPredicted token ids:", predictions.squeeze().tolist())
        # print("Predicted token:", itos[predictions.squeeze().item()])
        # print("Vocabulary size:", len(itos))
        
        # Decode and print
        completion = decode(output[0].tolist())
        print("Final output:", completion)

# After loading meta.pkl, add these debug prints:
# print("\nVocabulary check:")
# print("First 10 tokens in vocabulary:", [itos[i] for i in range(min(10, len(itos)))])
# print("Special tokens:", {
#     'MASK': stoi.get('[MASK]'),
#     'space': stoi.get(' '),
#     'newline': stoi.get('\n')
# })
