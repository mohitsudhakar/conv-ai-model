import numpy as np
import torch
from torch.utils.data import Dataset

class ConvAIDataset(Dataset):

    def parse(self):
        """
        Note: This parse function expects the no_cands (no candidates) version  of the ConvAI2 dataset.
        This means that this function would have to be modified for both_ (both personas) and original_ (with multiple candidates) versions.

        :return:
        """
        with open(self.filename, 'r') as f:
            chats = []
            lines = f.readlines()
            in_persona, in_dialog = False, False
            chat = {'dialog':[], 'persona':[]}
            for line in lines[:20]:
                line = line.strip()
                if len(line) == 0:
                    continue
                # if next persona has started, add current set to list
                if in_persona and in_dialog and 'your persona: ' in line:
                    # add curr to data
                    chats.append(chat)
                    chat = {'dialog':[], 'persona':[]}
                    in_persona, in_dialog = False, False

                if 'your persona: ' in line:
                    in_persona = True
                    text = line.split('your persona: ')[1]
                    chat['persona'].append(text)
                else:
                    in_dialog = True
                    idx = line.find(' ') + 1
                    text = line[idx:]
                    text = text.split('\t')
                    text = [t.strip() for t in text]
                    chat['dialog'].extend(text)

            # add the last set
            chats.append(chat)
        return chats

    def convert_to_bpe(self, data, bpe_vocab):
        bpe_data = []
        for chat in data:
            dialog, persona = chat['dialog'], chat['persona']
            dialog_toks = [bpe_vocab.string2ids(d) for d in dialog]
            persona_toks = [bpe_vocab.string2ids(p) for p in persona]

            # every input should have a response, so remove last one if number of utterances is odd
            if len(dialog_toks) % 2 == 1:
                dialog_toks = dialog_toks[:-1]
            bpe_data.append((dialog_toks, persona_toks))
        return bpe_data


    def __init__(self, filename, max_seq_len, bpe_vocab):

        self.filename = filename
        self.max_seq_len = max_seq_len
        self.vocab = bpe_vocab

        chats = self.parse()

        self.data = self.convert_to_bpe(chats, self.vocab)


    def __getitem__(self, index):

        dialog, persona = self.data[index]

        persona = sum(persona, [])
        persona = [self.vocab.info_bos_id] + persona[:self.max_seq_len-2] + [self.vocab.info_eos_id]

        x = []
        for i, toks in enumerate(dialog[:-1], 1):
            if i % 2 == 1:
                toks = [self.vocab.talker1_bos_id] + toks + [self.vocab.talker1_eos_id]
            else:
                toks = [self.vocab.talker2_bos_id] + toks + [self.vocab.talker2_eos_id]
            x.extend(toks)
        x = x[-self.max_seq_len:]

        y = [self.vocab.bos_id] + dialog[-1] + [self.vocab.eos_id]
        y = y[:self.max_seq_len]

        return x, y, persona


    def __len__(self):
        return len(self.data)
