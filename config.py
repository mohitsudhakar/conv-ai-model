def get():
    return {
        'max_seq_len': 512,
        'train_data': 'data/train_self_revised_no_cands.txt',
        'bpe_vocab_path': 'vocab/bpe.vocab',
        'bpe_codes_path': 'vocab/bpe.code',
    }