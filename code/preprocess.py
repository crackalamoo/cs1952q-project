import pickle

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torchvision
from torchvision import transforms
from datasets import load_dataset
import spacy
import gzip
import shutil
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset")
    parser.add_argument("--vocab_size", default=4096, type=int)
    args = parser.parse_args()
    vocab_size = args.vocab_size


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def pickle_mnist():
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
    trainset.data = trainset.data.tolist()
    trainset = {b'data': trainset.data, b'labels': trainset.targets}
    with open('../data/mnist_train', 'wb') as fo:
        pickle.dump(trainset, fo)
    
    testset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
    testset.data = testset.data.tolist()
    testset = {b'data': testset.data, b'labels': testset.targets}
    with open('../data/mnist_test', 'wb') as fo:
        pickle.dump(testset, fo)
    
    shutil.rmtree('./data')

def get_image_classifier_data(file_path, classes=None, num_channels=3, image_size=32, batch_size=64, shuffle=True):
    unpickled_file = unpickle(file_path)
    inputs = unpickled_file[b'data']
    labels = unpickled_file[b'labels']

    inputs = np.asarray(inputs)
    labels = np.asarray(labels)

    if classes is not None:
        indexing = (labels == classes[0])
        for i in range(1, len(classes)):
            indexing |= (labels == classes[i])
        inputs = inputs[indexing]
        labels = labels[indexing]
        vfunc = np.vectorize(lambda l: classes.index(l))
        labels = vfunc(labels)

    inputs = torch.tensor(inputs, dtype=torch.float32)
    labels = torch.nn.functional.one_hot(torch.tensor(labels), num_classes=(10 if classes is None else len(classes)))
    idxs = torch.arange(0, inputs.size(0))

    # Reshape and transpose images to match PyTorch's convention (num_inputs, num_channels, width, height)
    inputs = inputs.view(-1, num_channels, image_size, image_size)

    # Normalize inputs
    inputs /= 255.0

    dataset = TensorDataset(inputs, labels, idxs)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return data_loader

def pickle_wmt(num_wmt_samples=30000, num_30k_samples=30000):
    en_tok = spacy.load('en_core_web_sm')
    fr_tok = spacy.load('fr_core_news_sm')

    dataset = load_dataset('wmt/wmt14', data_dir='fr-en', split='train', streaming=True)
    english_spacy = []
    french_spacy = []
    en_toks = {}
    fr_toks = {}
    en_freqs = {}
    fr_freqs = {}

    def add_sentence(sentence, freqs, arr, tok):
        sentence = tok(sentence.lower())
        for token in sentence:
            freqs[token.text] = freqs.get(token.text, 0) + 1
        arr.append(sentence)


    for i, row in enumerate(iter(dataset)):
        if i == num_wmt_samples:
            break
        translation = row['translation']
        add_sentence(translation['en'], en_freqs, english_spacy, en_tok)
        add_sentence(translation['fr'], fr_freqs, french_spacy, fr_tok)
    print("Finished tokenizing WMT14")

    url_base = 'https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/'
    train_urls = ('train.en.gz', 'train.fr.gz')
    val_urls = ('val.en.gz', 'val.fr.gz')

    train_filepaths = []
    val_filepaths = []
    import urllib
    for url in train_urls:
        train_filepath = '../data/'+url.split('/')[-1]
        with urllib.request.urlopen(url_base + url) as response, open(train_filepath, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
        train_filepaths.append(train_filepath)
    for url in val_urls:
        val_filepath = '../data/'+url.split('/')[-1]
        with urllib.request.urlopen(url_base + url) as response, open(val_filepath, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
        val_filepaths.append(val_filepath)
    with gzip.open(train_filepaths[0], 'rb') as f:
        train_en = f.readlines()
        for i, line in enumerate(train_en):
            if i == num_30k_samples:
                break
            en_i = line.decode('utf-8')
            add_sentence(en_i, en_freqs, english_spacy, en_tok)
    with gzip.open(train_filepaths[1], 'rb') as f:
        train_fr = f.readlines()
        for i, line in enumerate(train_fr):
            if i == num_30k_samples:
                break
            fr_i = line.decode('utf-8')
            add_sentence(fr_i, fr_freqs, french_spacy, fr_tok)
    print("Finished tokenizing Multi30k")

    en_freqs = sorted(en_freqs.items(), key=lambda x: x[1], reverse=True)
    fr_freqs = sorted(fr_freqs.items(), key=lambda x: x[1], reverse=True)
    print(f'English vocab size: {len(en_freqs)}')
    print(f'French vocab size: {len(fr_freqs)}')

    fr_vocab_prop = 0.0
    for i in range(vocab_size):
        fr_vocab_prop += fr_freqs[i][1]
    fr_vocab_prop /= sum([x[1] for x in fr_freqs])
    print(f'French vocab proportion: {fr_vocab_prop}')

    for i in range(vocab_size-4):
        en_toks[en_freqs[i][0]] = i+4
        fr_toks[fr_freqs[i][0]] = i+4
    en_toks['<pad>'] = 0
    fr_toks['<pad>'] = 0
    en_toks['<bos>'] = 1
    fr_toks['<bos>'] = 1
    en_toks['<eos>'] = 2
    fr_toks['<eos>'] = 2
    en_toks['<unk>'] = 3
    fr_toks['<unk>'] = 3


    english = []
    french = []
    for i in range(num_wmt_samples + num_30k_samples):
        en_i = []
        fr_i = []
        for token in english_spacy[i]:
            if token.text not in en_toks:
                en_i.append(en_toks['<unk>'])
            else:
                en_i.append(en_toks[token.text])
        for token in french_spacy[i]:
            if token.text not in fr_toks:
                fr_i.append(fr_toks['<unk>'])
            else:
                fr_i.append(fr_toks[token.text])
        english.append(en_i)
        french.append(fr_i)
    print("Finished creating training set")

    trainset = {b'data': english, b'labels': french, b'data_tok': en_toks, b'labels_tok': fr_toks}
    with open('../data/wmt_train', 'wb') as fo:
        pickle.dump(trainset, fo)
    print("Finished pickling training set")

    dataset = load_dataset('wmt/wmt14', data_dir='fr-en', split='validation', streaming=True)
    english = []
    french = []
    def get_tokens(sentence, tok):
        sentence = tok(sentence.lower())
        res = []
        for token in sentence:
            if token.text in en_toks:
                res.append(en_toks[token.text])
            else:
                res.append(en_toks['<unk>'])
        return res
    for i, row in enumerate(iter(dataset)):
        if i == num_wmt_samples:
            break
        translation = row['translation']
        en_i = get_tokens(translation['en'], en_tok)
        fr_i = get_tokens(translation['fr'], fr_tok)
        english.append(en_i)
        french.append(fr_i)
    with gzip.open(val_filepaths[0], 'rb') as f:
        val_en = f.readlines()
        for i, line in enumerate(val_en):
            if i == num_30k_samples:
                break
            en_i = line.decode('utf-8')
            en_i = get_tokens(en_i, en_tok)
            english.append(en_i)
    with gzip.open(val_filepaths[1], 'rb') as f:
        val_fr = f.readlines()
        for i, line in enumerate(val_fr):
            if i == num_30k_samples:
                break
            fr_i = line.decode('utf-8')
            fr_i = get_tokens(fr_i, fr_tok)
            french.append(fr_i)

    print("Finished creating test set")
    testset = {b'data': english, b'labels': french}
    with open('../data/wmt_test', 'wb') as fo:
        pickle.dump(testset, fo)
    print("Finished pickling test set")

def collate_language_batch(batch_data):
    in_batch, out_batch, idx_batch = [], [], []
    make_sentence = lambda x: torch.cat((torch.tensor([1]), torch.tensor(x), torch.tensor([2])))
    for (input, label, idx) in batch_data:
        in_sentence = make_sentence(input)
        out_sentence = make_sentence(label)
        in_batch.append(in_sentence)
        out_batch.append(out_sentence)
        idx_tensor = torch.tensor(idx, dtype=torch.int32)
        idx_batch.append(idx_tensor)
    in_batch = torch.nn.utils.rnn.pad_sequence(in_batch, padding_value=0)
    out_batch = torch.nn.utils.rnn.pad_sequence(out_batch, padding_value=0)
    return in_batch, out_batch, idx_batch

def get_language_model_data(file_path, batch_size=64, shuffle=True, include_tok=False):
    """
    Given a file path, returns an array of input sequences and an array of label sequences.
    """

    unpickled_file = unpickle(file_path)
    inputs = unpickled_file[b'data']
    labels = unpickled_file[b'labels']

    dataset = list(zip(inputs, labels, range(len(inputs))))
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_language_batch)
    
    if include_tok:
        return data_loader, unpickled_file[b'data_tok'], unpickled_file[b'labels_tok']
    return data_loader

if __name__ == '__main__':
    if args.dataset == 'mnist':
        pickle_mnist()
    elif args.dataset == 'wmt':
        pickle_wmt()