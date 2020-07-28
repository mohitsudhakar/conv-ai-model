from torch.utils.data.dataloader import DataLoader

import config
from dataloader import ConvAIDataset
from utils import combine_contexts
from vocab.text import BPEVocab

config = config.get()
max_seq_len = config['max_seq_len']
train_data = config['train_data']
bpe_vocab_path = config['bpe_vocab_path']
bpe_codes_path = config['bpe_codes_path']
vocab = BPEVocab.from_files(vocab_path=bpe_vocab_path, codes_path=bpe_codes_path)
params = {'batch_size': 64, 'shuffle': True, 'num_workers': 1, 'collate_fn': combine_contexts}

if __name__ == '__main__':

    dataset = ConvAIDataset(filename=train_data,
                            max_seq_len=max_seq_len,
                            bpe_vocab=vocab)

    dataloader = DataLoader(dataset, **params)

    

    for i, (contexts, targets) in enumerate(dataloader):
        print(i, contexts, targets)
        exit(0)

