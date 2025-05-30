import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import re

torch.manual_seed(7)
df = pd.read_csv('data/eng_-french.csv')
# df = df[:500]

def tokenize_with_punctuation(text):
    # splits on words and punctuation, keeping both
    return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)

def E_int_tokenize(text):
    text = tokenize_with_punctuation(text)
    return [ E_w_to_i.get(word) for word in text]

def F_int_tokenize(text):
    text = tokenize_with_punctuation(text)
    return [ F_w_to_i.get(word) for word in text]

# Vocab for English
english_sentence = " ".join(df['English words/sentences'])
english_tokens = tokenize_with_punctuation(english_sentence)

E_w_to_i = {k:i+3 for i, k in enumerate(set(english_tokens))}
E_w_to_i ['<pad>'] = 0
E_w_to_i ['<sos>'] = 1
E_w_to_i ['<eos>'] = 2
E_i_to_w = {i: k for k, i in E_w_to_i.items()}

E_vocab_size = len(E_i_to_w.values())

# Vocab for French
french_sentence = " ".join(df['French words/sentences'])
french_tokens = tokenize_with_punctuation(french_sentence)

F_w_to_i = {k:i+3 for i, k in enumerate(set(french_tokens))}
F_w_to_i['<pad>'] = 0
F_w_to_i['<sos>'] = 1
F_w_to_i['<eos>'] = 2
F_i_to_w = {i: k for k, i in F_w_to_i.items()}

F_vocab_size = len(F_i_to_w.values())

# 
df['E_int_tokenize'] = df['English words/sentences'].apply(E_int_tokenize)
df['F_int_tokenize'] = df['French words/sentences'].apply(F_int_tokenize)

# Padding
# def collate_fn(batch):
#     src_batch, tgt_batch = zip(*batch)
#     src_padded = pad_sequence(src_batch, batch_first=True, padding_value=E_w_to_i['<pad>'])
#     tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=F_w_to_i['<pad>'])
#     return src_padded, tgt_padded
# This cause last error coz i am padding both seprately

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)

    # Find max length across both source and target sequences
    max_len = max(max(seq.size(0) for seq in src_batch), max(seq.size(0) for seq in tgt_batch))
    # Pad source and target to the same length
    src_padded = pad_sequence([torch.cat([seq, torch.full((max_len - seq.size(0),), E_w_to_i['<pad>'], dtype=seq.dtype)]) for seq in src_batch], batch_first=True)
    tgt_padded = pad_sequence([torch.cat([seq, torch.full((max_len - seq.size(0),), F_w_to_i['<pad>'], dtype=seq.dtype)]) for seq in tgt_batch], batch_first=True)

    return src_padded, tgt_padded


class translation_dataset(Dataset):
    def __init__(self, X, y):
        super().__init__()
        self.X = X # English
        self.y = y # French

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        src = [E_w_to_i['<sos>']] + self.X.iloc[index] + [E_w_to_i['<eos>']] 
        target = [F_w_to_i['<sos>']] + self.y.iloc[index] + [F_w_to_i['<eos>']] 
        return torch.tensor(src, dtype=torch.long), torch.tensor(target, dtype=torch.long)
    
dataset = translation_dataset(df['E_int_tokenize'], df['F_int_tokenize'])
batched_dataset = DataLoader(dataset=dataset, batch_size=36, shuffle= True, drop_last= True, collate_fn=collate_fn)

def get_batched_dataset():
    return batched_dataset

def get_vocab_sizes():
    return F_vocab_size, E_vocab_size

def get_word_index():
    return E_w_to_i, E_i_to_w, F_w_to_i, F_i_to_w