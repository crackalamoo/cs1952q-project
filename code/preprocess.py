import pickle

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torchvision
from torchvision import transforms
from datasets import load_dataset
import spacy
import shutil
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset")
    parser.add_argument("--vocab_size", default=512, type=int)
    args = parser.parse_args()
    vocab_size = args.vocab_size


def unpickle(file):
    """
    CIFAR data contains the files data_batch_1, data_batch_2, ..., 
    as well as test_batch. We have combined all train batches into one
    batch for you. Each of these files is a Python "pickled" 
    object produced with cPickle. The code below will open up a 
    "pickled" object (each file) and return a dictionary.

    :param file: the file to unpickle
    :return: dictionary of unpickled data
    """
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
    """
    Given a file path and two target classes, returns an array of 
    normalized inputs (images) and an array of labels.

    :return: normalized NumPy array of inputs and tensor of labels, where 
    inputs are of type np.float32 and has size (num_inputs, width, height, num_channels) and labels 
    has size (num_examples, num_classes)
    """
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

def pickle_wmt(num_train_samples=3000):
    en_tok = spacy.load('en_core_web_sm')
    fr_tok = spacy.load('fr_core_news_sm')

    dataset = load_dataset('wmt/wmt14', data_dir='fr-en', split='train', streaming=True)
    english_spacy = []
    french_spacy = []
    en_toks = {}
    fr_toks = {}
    en_freqs = {}
    fr_freqs = {}
    for i, row in enumerate(iter(dataset)):
        translation = row['translation']
        en_i = en_tok(translation['en'].lower())
        fr_i = fr_tok(translation['fr'].lower())
        for token in en_i:
            en_freqs[token.text] = en_freqs.get(token.text, 0) + 1
        for token in fr_i:
            fr_freqs[token.text] = fr_freqs.get(token.text, 0) + 1
        english_spacy.append(en_i)
        french_spacy.append(fr_i)
        if i == num_train_samples-1:
            break
    
    en_freqs = sorted(en_freqs.items(), key=lambda x: x[1], reverse=True)
    fr_freqs = sorted(fr_freqs.items(), key=lambda x: x[1], reverse=True)
    for i in range(vocab_size-3):
        en_toks[en_freqs[i][0]] = i+3
        fr_toks[fr_freqs[i][0]] = i+3
    en_toks['<pad>'] = 0
    fr_toks['<pad>'] = 0
    en_toks['<bos>'] = 1
    fr_toks['<bos>'] = 1
    en_toks['<eos>'] = 2
    fr_toks['<eos>'] = 2
    en_toks['<unk>'] = vocab_size
    fr_toks['<unk>'] = vocab_size

    
    english = []
    french = []
    for i in range(num_train_samples):
        en_i = []
        fr_i = []
        for token in english_spacy[i]:
            if token.text not in en_toks:
                en_i.append(len(en_toks))
            else:
                en_i.append(en_toks[token.text])
        for token in french_spacy[i]:
            if token.text not in fr_toks:
                fr_i.append(len(fr_toks))
            else:
                fr_i.append(fr_toks[token.text])
        english.append(en_i)
        french.append(fr_i)

    trainset = {b'data': english, b'labels': french, b'data_tok': en_toks, b'labels_tok': fr_toks}
    with open('../data/wmt_train', 'wb') as fo:
        pickle.dump(trainset, fo)

    dataset = load_dataset('wmt/wmt14', data_dir='fr-en', split='validation', streaming=True)
    english = []
    french = []
    for i, row in enumerate(iter(dataset)):
        translation = row['translation']
        en_tokenized = en_tok(translation['en'].lower())
        fr_tokenized = fr_tok(translation['fr'].lower())
        en_i = []
        fr_i = []
        for token in en_tokenized:
            if token.text not in en_toks:
                en_i.append(len(en_toks))
            else:
                en_i.append(en_toks[token.text])
        for token in fr_tokenized:
            if token.text not in fr_toks:
                fr_i.append(len(fr_toks))
            else:
                fr_i.append(fr_toks[token.text])
        english.append(en_i)
        french.append(fr_i)
    testset = {b'data': english, b'labels': french}
    with open('../data/wmt_test', 'wb') as fo:
        pickle.dump(testset, fo)

def get_language_model_data(file_path, batch_size=64, shuffle=True, include_tok=False):
    """
    Given a file path, returns an array of input sequences and an array of label sequences.
    """

    unpickled_file = unpickle(file_path)
    inputs = unpickled_file[b'data']
    labels = unpickled_file[b'labels']
    example_idx = 0

    def collate_batch(batch_data):
        nonlocal example_idx
        in_batch, out_batch, idx_batch = [], [], []
        make_sentence = lambda x: torch.cat((torch.tensor([1]), torch.tensor(x), torch.tensor([2])))
        for (input, label) in batch_data:
            in_sentence = make_sentence(input)
            out_sentence = make_sentence(label)
            in_batch.append(in_sentence)
            out_batch.append(out_sentence)
            idx_tensor = torch.tensor(example_idx, dtype=torch.int32)
            idx_batch.append(idx_tensor)
            example_idx += 1
        in_batch = torch.nn.utils.rnn.pad_sequence(in_batch, padding_value=0)
        out_batch = torch.nn.utils.rnn.pad_sequence(out_batch, padding_value=0)
        return in_batch, out_batch, idx_batch

    dataset = list(zip(inputs, labels))
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_batch)
    
    if include_tok:
        return data_loader, unpickled_file[b'data_tok'], unpickled_file[b'labels_tok']
    return data_loader

if __name__ == '__main__':
    if args.dataset == 'mnist':
        pickle_mnist()
    elif args.dataset == 'wmt':
        pickle_wmt()