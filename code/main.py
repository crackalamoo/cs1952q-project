from __future__ import absolute_import

import argparse
import time
from collections import defaultdict
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from cifar10 import Cifar10Model, get_cifar10_data
from mnist import MNISTModel, get_mnist_data
from wmt import WMTModel, get_wmt_data, test_translate_callback
from preprocess import collate_language_batch
from torch.utils.data import DataLoader, Subset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--model", default="wmt")
    parser.add_argument('--samp', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--samp-prop", type=float, default=0.35)
    parser.add_argument("--reintroduce", type=float, default=0.2)
    parser.add_argument("--store-loss", type=int, default=2)
    parser.add_argument("--runs", type=int, default=1)
    args = parser.parse_args()

    EPOCHS = args.epochs
    device = args.device
    MODEL = args.model
    USE_SAMPLING = args.samp
    LOSS_THRESHOLD = 0.9
    LOSS_THRESHOLD_VELOCITY = 0
    FORCE_PROPORTION = args.samp_prop
    REINTRODUCE_LEARNED = args.reintroduce
    STORED_LOSSES = args.store_loss
    STOP_SAMPLING = 20
    LR = 5e-5
    RUNS = args.runs


def load_sampling(train_loader, sample_losses, epoch, batch_size=None, collate_fn=None):
    if batch_size is None:
        batch_size = train_loader.batch_size

    dataset = train_loader.dataset
    if FORCE_PROPORTION is None:
        threshold = LOSS_THRESHOLD - LOSS_THRESHOLD_VELOCITY*epoch
    else:
        all_losses = np.zeros(len(sample_losses))
        for i, idx in enumerate(sample_losses):
            all_losses[i] = np.mean(sample_losses[idx])
        threshold = np.quantile(all_losses, 1-FORCE_PROPORTION)
    keep_idx = set()
    for idx in sample_losses:
        if np.mean(sample_losses[idx]) > threshold or np.random.random() < REINTRODUCE_LEARNED:
            keep_idx.add(idx)
    subset = Subset(dataset, list(keep_idx))
    proportion = len(keep_idx)/(len(train_loader)*batch_size)
    print("Proportion kept:", proportion)

    training = DataLoader(subset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    return training, proportion


def train(model, train_loader, val_loader=None, epochs=EPOCHS, use_sampling=USE_SAMPLING, lr=LR,
          use_labels_as_input=False, grad_clip=False, collate_fn=None, callback=None):
    losses = []
    accuracies = []
    val_losses = []
    val_accuracies = []
    training = train_loader
    sample_losses = defaultdict(lambda: np.full(STORED_LOSSES, np.inf))
    times = []
    proportions = []
    start_time = time.time()
    rebatch_time = 0.0
    val_time = 0.0
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.to(device)
        accuracy = []
        epoch_losses = []
        if use_sampling and epoch >= STORED_LOSSES and epoch < STOP_SAMPLING:
            start_load = time.time()
            training, proportion = load_sampling(
                train_loader, sample_losses, epoch,
                collate_fn=collate_fn)
            rebatch_time += time.time() - start_load
            proportions.append(proportion)
        else:
            training = train_loader
            proportions.append(1)
        for batch_no, (inputs, labels, idxs) in enumerate(training):
            model.train()
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            if use_labels_as_input:
                outputs = model(inputs, labels)
            else:
                outputs = model(inputs)
            
            batch_losses = model.batch_losses(outputs, labels)
            for idx, loss_i in zip(idxs, batch_losses):
                idx = idx.item()
                sample_losses[idx] = np.roll(sample_losses[idx], -1)
                sample_losses[idx][-1] = loss_i.item()
            loss = torch.mean(batch_losses)
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            epoch_losses.append(loss.item())
            if not isinstance(model, WMTModel) or batch_no < 2:
                # 2 batches times 64 sample sentences per batch is 128 sample sentences for validation
                start_val = time.time()
                model.eval()
                acc_i = model.accuracy(outputs, labels)
                val_time += time.time() - start_val
                accuracy.append(acc_i)
        train_acc = np.mean(accuracy)
        accuracies.append(train_acc)
        losses.append(np.mean(epoch_losses))
        if val_loader is not None:
            val_loss, val_acc = test(model, val_loader,
                                     use_labels_as_input=use_labels_as_input,
                                     max_batch=2 if isinstance(model, WMTModel) else None)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            print(
                f"Epoch: {epoch}, Train Acc: {train_acc}, Train Loss: {np.mean(epoch_losses)}, Val Acc: {val_acc}, Val Loss: {val_loss}")
        else:
            print(f"Epoch: {epoch}, Training Accuracy: {train_acc}")
        times.append(time.time() - start_time)
        print(f"Rebatch time: {rebatch_time} s")
        print(f"Validation time: {val_time} s")
        print(f"Total time: {times[-1]} s")
        if callback is not None:
            callback(model, epoch)
        if device == 'cuda':
            torch.cuda.empty_cache()
        sys.stdout.flush()
    if val_loader is not None:
        return {
            'loss': losses, 'acc': accuracies, 'val_loss': val_losses, 'val_acc': val_accuracies,
            'times': times, 'prop': proportions, 'rebatch_time': rebatch_time, 'val_time': val_time
        }
    return losses, accuracies, times


def test(model, test_loader, use_labels_as_input=False, max_batch=None):
    model.to("cpu")
    model.eval()
    losses = []
    accuracy = []
    with torch.no_grad():
        for batch_no, (inputs, labels, idxs) in enumerate(test_loader):
            inputs, labels = inputs, labels
            if use_labels_as_input:
                outputs = model(inputs, labels)
            else:
                outputs = model(inputs)
            losses.append(model.loss(outputs, labels))
            accuracy.append(model.accuracy(outputs, labels))
            if max_batch is not None and batch_no >= max_batch:
                break
    test_acc = np.mean(accuracy)
    test_loss = np.mean(losses)
    return test_loss, test_acc


def do_run(ModelClass, get_data, run_no=0, grad_clip=False, collate_fn=None):
    train_loader, test_loader, extras = get_data()

    model = ModelClass()
    is_translate = isinstance(model, WMTModel)
    if is_translate:
        model.set_data_tok(extras['data_tok'])
        model.set_labels_tok(extras['labels_tok'])
    for p in model.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p)

    
    train_res = train(model, train_loader, val_loader=test_loader,
                      use_labels_as_input=is_translate, epochs=EPOCHS,
                      collate_fn=collate_fn, grad_clip=grad_clip,
                      callback=test_translate_callback if is_translate else None)
    train_loss, train_acc, val_loss, val_acc, train_times, proportions, rebatch_time, val_time = (
        train_res['loss'], train_res['acc'], train_res['val_loss'], train_res['val_acc'],
        train_res['times'], train_res['prop'], train_res['rebatch_time'], train_res['val_time'])

    test_loss, test_acc = test(model, test_loader, use_labels_as_input=is_translate)

    print(f"Final test accuracy: {test_acc}")
    print(f"Rebatch time: {rebatch_time} s")
    print(f"Validation time: {val_time} s")
    print(f"Total time: {train_times[-1]} s")

    fname = 'samp' if USE_SAMPLING else 'reg'
    with open(f'../results/{fname}.npy', 'wb+' if run_no == 0 else 'ab+') as f:
        np.save(f, train_times)
        np.save(f, val_loss)
        np.save(f, val_acc)
    

def do_graph():
    def get_stats(fname):
        total_times = []
        total_loss = []
        total_acc = []
        with open(f'../results/{fname}.npy', 'rb') as f:
            for i in range(RUNS):
                reg_times = np.load(f)
                reg_loss = np.load(f)
                reg_acc = np.load(f)
                total_times.append(reg_times)
                total_loss.append(reg_loss)
                total_acc.append(reg_acc)
        new_times = np.linspace(np.min(total_times), np.max(total_times), 1000)
        def get_extrapolated_values(times, values):
            new_values = np.interp(new_times, times, values)
            return new_values
        total_loss = [get_extrapolated_values(reg_times, reg_loss) for reg_times,reg_loss in zip(total_times, total_loss)]
        total_acc = [get_extrapolated_values(reg_times, reg_acc) for reg_times,reg_acc in zip(total_times, total_acc)]
        total_loss = np.array(total_loss)
        total_acc = np.array(total_acc)
        mean_loss = np.mean(total_loss, axis=0)
        mean_acc = np.mean(total_acc, axis=0)
        if total_loss.shape[0] == 1:
            return new_times, mean_loss, mean_acc, 0, 0, 0, 0
        stdev_loss = np.std(total_loss, axis=0, ddof=1)
        stdev_acc = np.std(total_acc, axis=0, ddof=1)
        stderr_loss = stdev_loss/np.sqrt(total_loss.shape[0])
        stderr_acc =stdev_acc/np.sqrt(total_acc.shape[0])
        return new_times, mean_loss, mean_acc, stdev_loss, stdev_acc, stderr_loss, stderr_acc
    
    reg_times, reg_loss, reg_acc, reg_stdev_loss, reg_stdev_acc, reg_stderr_loss, reg_stderr_acc = get_stats('reg')
    if USE_SAMPLING:
        cur_times, cur_loss, cur_acc, cur_stdev_loss, cur_stdev_acc, cur_stderr_loss, cur_stderr_acc = get_stats('samp')


    plt.rcParams.update({'font.size': 16})
    plt.figure(figsize=(5.6,4.2), dpi=200)
    if USE_SAMPLING:
        plt.plot(cur_times, cur_acc, label='sampling', color='tab:orange')
        # plt.fill_between(cur_times, cur_acc-cur_stderr_acc, cur_acc+cur_stderr_acc, alpha=0.3, color='tab:orange')
        plt.fill_between(cur_times, cur_acc-cur_stdev_acc, cur_acc+cur_stdev_acc, alpha=0.15, color='tab:orange')
    plt.plot(reg_times, reg_acc, label='no sampling', color='tab:blue')
    # plt.fill_between(reg_times, reg_acc-reg_stderr_acc, reg_acc+reg_stderr_acc, alpha=0.3, color='tab:blue')
    plt.fill_between(reg_times, reg_acc-reg_stdev_acc, reg_acc+reg_stdev_acc, alpha=0.15, color='tab:blue')
    # plt.plot(cur_times[STORED_LOSSES:],
    #          proportions[STORED_LOSSES:], label='proportion')
    plt.xlabel('Time (s)')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('../results/cur_vs_reg.png', bbox_inches='tight')

    plt.figure(figsize=(5.6,4.2), dpi=200)
    if USE_SAMPLING:
        plt.plot(cur_times, cur_loss, label='sampling', color='tab:orange')
        # plt.fill_between(cur_times, cur_loss-cur_stderr_loss, cur_loss+cur_stderr_loss, alpha=0.3, color='tab:orange')
        plt.fill_between(cur_times, cur_loss-cur_stdev_loss, cur_loss+cur_stdev_loss, alpha=0.15, color='tab:orange')
    plt.plot(reg_times, reg_loss, label='no sampling', color='tab:blue')
    # plt.fill_between(reg_times, reg_loss-reg_stderr_loss, reg_loss+reg_stderr_loss, alpha=0.3, color='tab:blue')
    plt.fill_between(reg_times, reg_loss-reg_stdev_loss, reg_loss+reg_stdev_loss, alpha=0.15, color='tab:blue')
    plt.xlabel('Time (s)')
    plt.ylabel('Cross-entropy loss')
    plt.legend()
    plt.savefig('../results/cur_vs_reg_loss.png', bbox_inches='tight')
    plt.show()

def main():
    model_class = {
        'mnist': MNISTModel,
        'cifar10': Cifar10Model,
        'wmt': WMTModel
    }[MODEL]
    data_func = {
        'mnist': get_mnist_data,
        'cifar10': get_cifar10_data,
        'wmt': get_wmt_data
    }[MODEL]
    collate_func = {
        'mnist': None,
        'cifar10': None,
        'wmt': collate_language_batch
    }[MODEL]
    grad_clip = {
        'mnist': False,
        'cifar10': False,
        'wmt': True
    }[MODEL]
    for i in range(RUNS):
        torch.manual_seed(42+i)
        do_run(model_class, data_func, i, grad_clip=grad_clip, collate_fn=collate_func)
    try:
        do_graph()
    except FileNotFoundError:
        print("Missing data for graphing")

if __name__ == '__main__':
    main()
