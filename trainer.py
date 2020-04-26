
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR

from our_gpt2.gpt2_transformer import GPT2LMHeadModel
from dataloader import ConvAIDataset
from utils import combine_contexts
from vocab.text import BPEVocab


train_data = 'data/train_self_revised_no_cands.txt'
bpe_vocab_path = 'vocab/bpe.vocab'
bpe_codes_path = 'vocab/bpe.code'
params = {'batch_size': 16, 'shuffle': True, 'num_workers': 2}

cuda = True
learning_rate = 1e-3
max_seq_len = 128

vocab_size = 40000
hidden_dim = 128
embedding_dim=768
num_heads = 2
num_layers = 2
seq_len= 2*max_seq_len
dropout=0.1
num_epochs = 5

# Load data
vocab = BPEVocab.from_files(bpe_vocab_path, bpe_codes_path)
dataset = ConvAIDataset(filename=train_data,
                        max_seq_len=max_seq_len,
                        bpe_vocab=vocab)
dataloader = DataLoader(dataset, **params)

# Set device type
if cuda and torch.cuda.is_available():
    print("Running on GPU")
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("-" * 84)
print("Running on device type: {}".format(device))


# Initialize model, optimizer, scheduler
model = GPT2LMHeadModel(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            seq_len= seq_len,
            hidden_dim=hidden_dim,
            device=device,
            dropout=dropout)

# Check for multiple instances of GPU's and use them
if torch.cuda.device_count() > 1:
    print("Running on ", torch.cuda.device_count(), "GPU's")
    model = nn.DataParallel(model)
    model.to(device)
else:
    model.to(device)

optimizer = torch.optim.Adam(lr=learning_rate, params=model.parameters())
scheduler = ReduceLROnPlateau(optimizer, "min", patience=10, verbose=True,)

print("-" * 84)
print("Start Training")
start_time = time.time()
training_loss_list = []

for epoch in range(num_epochs):
    epoch_start = time.time()
    model.train()
    training_loss = 0
    for i, (prev_context, target, persona) in enumerate(dataloader):
        prev_context, targets, persona = torch.stack(prev_context), torch.stack(target), torch.stack(persona)
        prev_context, targets, persona = torch.t(prev_context), torch.t(target), torch.t(persona) 
        context = torch.cat([persona, prev_context], dim=1)
        context = F.pad(input=context, pad=(0, 2*max_seq_len - context.size(1)), mode="constant", value=0)
        #print(i, context.shape, target.shape)
        tokens_tensor =  context           
        for ind in range(target.shape[1]):
            optimizer.zero_grad()
            label = target[:,ind]
            #print(i, context.shape, target.shape)
            out = model(tokens_tensor)
            out = out[0]
            predictions = torch.softmax(out[:, -1, :], dim = 0)
            predictions = torch.log(predictions)
            loss = F.nll_loss(predictions, torch.tensor(label))
            print(loss)
            training_loss += loss.item()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()
            predicted_index = torch.argmax(predictions, dim = 1)
            tokens_tensor = torch.cat((tokens_tensor, label.unsqueeze(1)), dim = 1)
    print("epoch", epoch, "training_loss_per_epoch", training_loss)
    training_loss_list.append(training_loss)
    sys.exit()