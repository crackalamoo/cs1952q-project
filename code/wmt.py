import torch
import numpy as np
from preprocess import get_language_model_data
import spacy
import heapq

# https://pytorch.org/tutorials/beginner/translation_transformer.html#seq2seq-network-using-transformer

class PositionalEncoding(torch.nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * np.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = torch.nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: torch.Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

class TokenEmbedding(torch.nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: torch.Tensor):
        return self.embedding(tokens.long()) * np.sqrt(self.emb_size)

class WMTModel(torch.nn.Module):
    def __init__(self,
                 num_encoder_layers: int = 3,
                 num_decoder_layers: int = 3,
                 emb_size: int = 512,
                 nhead: int = 8,
                 vocab_size: int = 1024,
                 dim_feedforward: int = 512,
                 dropout: float = 0.0):
        super(WMTModel, self).__init__()
        self.transformer = torch.nn.Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)
        self.generator = torch.nn.Linear(emb_size, vocab_size)
        self.src_tok_emb = TokenEmbedding(vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)
        self.data_tok = None
        self.labels_tok = None

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, reduce_tgt: bool = True):
        pad_mask = lambda t: (t == 0).transpose(0, 1).float()
        if reduce_tgt:
            tgt_input = tgt[:-1, :] # exclude last token, which must be predicted
        else:
            tgt_input = tgt
        src_padding_mask = pad_mask(src)
        tgt_padding_mask = pad_mask(tgt_input)
        tgt_mask = (torch.triu(torch.ones((tgt_input.size(0), tgt_input.size(0)), device=tgt.device)) == 1).transpose(0, 1)
        tgt_mask = tgt_mask.float().masked_fill(tgt_mask == 0, float('-inf')).masked_fill(tgt_mask == 1, float(0.0))
        src_mask = torch.zeros(src.size(0), src.size(0), device=src.device).float()

        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt_input))
        outs = self.transformer(src_emb, tgt_emb,
                                src_mask=src_mask, tgt_mask=tgt_mask,
                                src_key_padding_mask=src_padding_mask,
                                tgt_key_padding_mask=tgt_padding_mask,
                                memory_key_padding_mask=src_padding_mask)
        logits = self.generator(outs)
        return logits

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor):
        return self.transformer.encoder(self.positional_encoding(
                            self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: torch.Tensor):
        return self.transformer.decoder(self.positional_encoding(
                          self.tgt_tok_emb(tgt)), memory,
                          tgt_mask)
    
    def loss(self, logits, labels):
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)
        logits = logits.transpose(0,1).transpose(1,2)
        labels = labels[1:, :].transpose(0,1) # exclude first token, which is only used as input
        res = loss_fn(logits, labels)
        return res

    def batch_losses(self, logits, labels):
        loss_fn = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=0)
        logits = logits.transpose(0,1).transpose(1,2)
        labels = labels[1:, :].transpose(0,1)
        res = loss_fn(logits, labels)
        res = torch.mean(res, dim=1)
        return res
    
    def accuracy(self, logits, labels):
        # predicted = torch.argmax(logits, dim=-1)
        # correct = labels[1:, :]
        # correct_predictions = ((predicted == correct) * (correct != 0)).sum().item()
        # total = (correct != 0).sum().item()
        # return correct_predictions / total

        # BLEU score computation
        def get_clip_seq(seq):
            eos_idx = self.data_tok['<eos>']
            pad_idx = self.data_tok['<pad>']
            if eos_idx in seq:
                if pad_idx in seq:
                    return seq[:min(seq.index(eos_idx)+1, seq.index(pad_idx)+1)]
                else:
                    return seq[:seq.index(eos_idx)+1]
            elif pad_idx in seq:
                return seq[:seq.index(pad_idx)+1]
            return seq
        def get_ngram_counts(seq, n):
            clip_seq = get_clip_seq(seq)
            counts = {}
            for i in range(len(clip_seq)+1-n):
                sub = tuple(clip_seq[i:i+n])
                if not sub in counts:
                    counts[sub] = 0
                counts[sub] += 1
            return counts
        def get_precision_i(i):
            precision = 0
            wi = 0
            for j in range(len(list_candidate)):
                snt = list_candidate[j]
                ref = list_reference[j]
                snt_counts = get_ngram_counts(snt, i)
                ref_counts = get_ngram_counts(ref, i)
                for igram in snt_counts:
                    precision += min(snt_counts[igram], ref_counts[igram] if igram in ref_counts else 0)
                    wi += snt_counts[igram]
            if precision != 0:
                precision /= wi
            return precision

        bleu_iter = labels.transpose(0,1).tolist()
        if len(bleu_iter) > 20:
            bleu_iter = bleu_iter[:20]
        for label in bleu_iter:
            list_candidate = [self.generate_translation(torch.tensor(label, device=labels.device), use_tokens=True)]
            list_reference = [label]

            precision = 1.0
            for i in range(1, 4+1):
                precision *= get_precision_i(i)
            precision **= (1/4.0)
            ref_len = 0
            out_len = 0
            assert len(list_candidate) == len(list_reference)
            for i in range(len(list_candidate)):
                ref_len += len(get_clip_seq(list_reference[i]))
                out_len += len(get_clip_seq(list_candidate[i]))
            precision *= min(1, np.exp(1 - ref_len/out_len))
            return precision
    
    def set_data_tok(self, tok):
        self.data_tok = tok
    def set_labels_tok(self, tok):
        self.labels_tok = tok

    def generate_translation(self, sentence, use_tokens=False, max_len=70):
        if not use_tokens:
            inputs = convert_to_tokens(sentence, self.data_tok)
        else:
            inputs = sentence
        res = [self.labels_tok['<bos>']]
        tok = -1
        inputs = inputs.unsqueeze(1)
        while tok != self.labels_tok['<eos>'] and len(res) < max_len:
            res_tensor = torch.tensor(res, device=inputs.device).unsqueeze(1)
            outputs = self(inputs, res_tensor, reduce_tgt=False)
            outputs[:,:,self.labels_tok['<bos>']] = -np.inf
            outputs[:,:,self.labels_tok['<pad>']] = -np.inf
            outputs[:,:,self.labels_tok['<unk>']] = -np.inf
            tok = torch.argmax(outputs[-1, 0, :], dim=-1)
            res.append(tok.item())
        if not use_tokens:
            translation = tokens_to_string(torch.tensor(res), self.labels_tok)
        else:
            translation = res
        return translation

def convert_to_tokens(sentence, tok, tok_name='en_core_web_sm'):
    # tok is either data_tok or labels_tok, mapping from token text to index
    spacy_tok = spacy.load(tok_name)
    sentence = spacy_tok(sentence)
    return torch.cat([
        torch.tensor([tok['<bos>']]),
        torch.tensor([(tok[token.text] if token.text in tok else tok['<unk>']) for token in sentence]),
        torch.tensor([tok['<eos>']])
    ])

def tokens_to_string(tokens, tok):
    reverse_tok = {v: k for k, v in tok.items()}
    return ' '.join([(reverse_tok[token.item()] if token.item() in reverse_tok else '<unk>') for token in tokens])

def test_translate_callback(model: WMTModel, epoch):
    sentence = "i am the president of the european parliament."
    translation = model.generate_translation(sentence)
    print(translation)

    sentence = "i am a student."
    translation = model.generate_translation(sentence)
    print(translation)

    sentence = "\"There is no need for radical surgery when all you need to do is take an aspirin,\" said Randy Rentschler, the commission's director of legislation and public affairs."
    translation = model.generate_translation(sentence)
    print(translation)

def get_wmt_data():
    AUTOGRADER_TRAIN_FILE = '../data/wmt_train'
    AUTOGRADER_TEST_FILE = '../data/wmt_test'

    train_loader, data_tok, labels_tok = get_language_model_data(AUTOGRADER_TRAIN_FILE, include_tok=True)
    test_loader = get_language_model_data(AUTOGRADER_TEST_FILE)

    return train_loader, test_loader, {'data_tok': data_tok, 'labels_tok': labels_tok}