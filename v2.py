import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64  # how many independent sequences will we process in parallel?
block_size = 256  # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = "mps" if torch.backends.mps.is_available() else "cpu"
eval_iters = 200
n_embd = 384  # number of embedding dimensions 384/6=64 for each head
n_head = 6  # number of heads
n_layer = 6  # number of layers
dropout = 0.2
# -------------------------

# Reading and Inspecting the Data
with open("input.txt", "r", encoding="utf-8") as file:
    data = file.read()

# All unique characters in the dataset
chars = sorted(list(set(data)))
vocab_size = len(chars)
# create a mapping of unique characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [
    stoi[ch] for ch in s
]  # encoder: take a string and convert it into a list of integers
decode = lambda l: "".join(
    [itos[i] for i in l]
)  # decoder: take a list of integers and convert it into a string

# Lets now split the dataset into training and validation sets
data_ = torch.tensor(encode(data), dtype=torch.long)
n = int(0.9 * len(data_))  # 90% of the data for training and 10% for validation
train_data, valid_data = data_[:n], data_[n:]


# data loading
def get_batch(split):
    # generate a small batch of data of input x and targets y
    data = train_data if split == "train" else valid_data
    ix = torch.randint(
        len(data) - block_size, (batch_size,)
    )  # starting index for each sequence
    x = torch.stack([data[i : i + block_size] for i in ix])  # input data
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])  # target data
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    """one head of self-attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B, T, H)
        q = self.query(x)  # (B, T, H)
        # compute the attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C**-0.5  # (B, T, T)
        wei = wei.masked_fill(
            self.tril[:T, :T] == 0, float("-inf")
        )  # mask out the lower half of the matrix
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B, T, H)
        out = wei @ v  # (B, T, H)
        return out


class MultiHeadAttention(nn.Module):
    """a multi-head attention module"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(out)
        out = self.proj(out)
        return out

class FeedForward(nn.Module):
    """a simple linear layer followed by non-linearity"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """
    
    def __init__(self, n_embd, n_head):
        # n_embd: number of embedding dimensions, n_head: number of heads
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ff = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # self.sa_head = MultiHeadAttention(
        #     4, n_embd // 4
        # )  # i.e, 4 heads of 8-dimensional self-attention
        # self.ff_head = FeedForward(n_embd)
        
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (Batch, Time, Channels)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=device)
        )  # (Time, Channels)
        x = tok_emb + pos_emb  # (Batch, Time, Channels)
        # x = self.sa_head(x)  # (Batch, Time, Channels)
        # x = self.ff_head(x)  # (Batch, Time, Channels)
        x = self.blocks(x)
        logits = self.lm_head(x)  # (Batch, Time, Vocab Size)
        # (B, T, C) -> (B, C, T)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # becomes (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # becomes (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # becomes (B, T+1)
        return idx


model = BigramLanguageModel()
model = model.to(device)  # Move the model to the MPS device

# optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"Iteration(step) {iter}: Train loss: {losses['train']:.4f}, Val loss: {losses['val']:.4f}"
        )

    # sample a batch of data
    xb, yb = get_batch("train")

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
idx = torch.zeros((1, 1), dtype=torch.long).to(
    device
)  # Move the input tensor to the MPS device
print(decode(model.generate(idx=idx, max_new_tokens=1000)[0].tolist()))
