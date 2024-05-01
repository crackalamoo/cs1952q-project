import torch
import numpy as np
from preprocess import get_language_model_data

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
                 num_encoder_layers: int = 2,
                 num_decoder_layers: int = 2,
                 emb_size: int = 512,
                 nhead: int = 2,
                 vocab_size: int = 512,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(WMTModel, self).__init__()
        self.transformer = torch.nn.Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)
        self.generator = torch.nn.Linear(emb_size, vocab_size)
        self.src_tok_emb = TokenEmbedding(vocab_size+2, emb_size)
        self.tgt_tok_emb = TokenEmbedding(vocab_size+2, emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor):
        pad_mask = lambda t: (t == 0).transpose(0, 1)
        tgt_input = tgt[:-1, :] # exclude last word, which must be predicted
        src_padding_mask = pad_mask(src)
        tgt_padding_mask = pad_mask(tgt_input)
        tgt_mask = torch.ones(tgt_input.size()[0], tgt_input.size()[0])
        tgt_mask = torch.triu(tgt_mask, diagonal=1).bool()

        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt_input))
        outs = self.transformer(src_emb, tgt_emb, tgt_mask=tgt_mask,
                                src_key_padding_mask=src_padding_mask,
                                tgt_key_padding_mask=tgt_padding_mask)
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
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=513)
        return loss_fn(logits.view(-1, logits.size(-1)), labels[1:, :].view(-1))

    def batch_losses(self, logits, labels):
        loss_fn = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=513)
        logits = logits.transpose(0,1).transpose(1,2)
        labels = labels[1:, :].transpose(0,1)
        res = loss_fn(logits, labels)
        res = torch.mean(res, dim=1)
        return res
    
    def accuracy(self, logits, labels):
        predicted = torch.argmax(logits, dim=-1)
        correct = labels[1:, :]
        correct_predictions = ((predicted == correct) * (correct != 0)).sum().item()
        total = (correct != 0).sum().item()
        return correct_predictions / total

def get_wmt_data():
    AUTOGRADER_TRAIN_FILE = '../data/wmt_train'
    AUTOGRADER_TEST_FILE = '../data/wmt_test'

    train_loader, data_tok, labels_tok = get_language_model_data(AUTOGRADER_TRAIN_FILE, include_tok=True)
    test_loader = get_language_model_data(AUTOGRADER_TEST_FILE)

    return train_loader, test_loader, {'data_tok': data_tok, 'labels_tok': labels_tok}