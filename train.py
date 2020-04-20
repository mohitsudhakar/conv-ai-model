from torch.utils.data.dataloader import DataLoader

from dataloader import ConvAIDataset
from utils import combine_contexts
from vocab.text import BPEVocab

max_seq_len = 512
train_data = 'data/train_self_revised_no_cands.txt'
bpe_vocab_path = 'vocab/bpe.vocab'
bpe_codes_path = 'vocab/bpe.code'
params = {'batch_size': 64, 'shuffle': True, 'num_workers': 2, 'collate_fn': combine_contexts}

if __name__ == '__main__':

    vocab = BPEVocab.from_files(bpe_vocab_path, bpe_codes_path)

    dataset = ConvAIDataset(filename=train_data,
                            max_seq_len=max_seq_len,
                            bpe_vocab=vocab)

    dataloader = DataLoader(dataset, **params)

    for i, (contexts, targets) in enumerate(dataloader):
        print(i, contexts, targets)
        exit(0)
