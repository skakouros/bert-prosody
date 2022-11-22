import os
import sys
import errno
import glob
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils import data
import torch.optim as optim
import prosody_dataset
from prosody_dataset import Dataset
from prosody_dataset import load_embeddings, add_pos_to_dataset
from model import Bert, BertLSTM, LSTM, BertRegression, LSTMRegression, WordMajority, ClassEncodings, BertAllLayers
from argparse import ArgumentParser, BooleanOptionalAction, Namespace
import time
from sklearn.metrics import precision_recall_fscore_support, classification_report
from pprintpp import pprint as pp
from pathlib import Path
from transformers import logging


# set transformers verbosity level to error
logging.set_verbosity_error()


def config_parser() -> Namespace:
    # init argument parser
    parser = ArgumentParser(description='Prosody prediction')

    parser.add_argument('--datadir',
                        type=str,
                        default='./data')
    parser.add_argument('--train_set',
                        type=str,
                        choices=['train',
                                 'train_100',
                                 'train_360'],
                        default='train_360')
    parser.add_argument('--batch_size',
                        type=int,
                        default=32)
    parser.add_argument('--epochs',
                        type=int,
                        default=2)
    parser.add_argument('--model',
                        type=str,
                        choices=['BertUncased',
                                 'BertCased',
                                 'BertLSTM',
                                 'LSTM',
                                 'BiLSTM',
                                 'BertRegression',
                                 'LSTMRegression',
                                 'WordMajority',
                                 'ClassEncodings',
                                 'BertAllLayers'],
                        default='BertUncased')
    parser.add_argument('--nclasses',
                        type=int,
                        default=3)
    parser.add_argument('--hidden_dim',
                        type=int,
                        default=600)
    parser.add_argument('--embedding_file',
                        type=str,
                        default='embeddings/glove.840B.300d.txt')
    parser.add_argument('--layers',
                        type=int,
                        default=1)
    parser.add_argument('--save_path',
                        type=str,
                        default='results.txt')
    parser.add_argument('--log_every',
                        type=int,
                        default=10)
    parser.add_argument('--learning_rate',
                        type=float,
                        default=0.00005)
    parser.add_argument('--weight_decay',
                        type=float,
                        default=0)
    parser.add_argument('--gpu',
                        type=int,
                        default=None)
    parser.add_argument('--fraction_of_train_data',
                        type=float,
                        default=1)
    parser.add_argument("--optimizer",
                        type=str,
                        choices=['rprop',
                                 'adadelta',
                                 'adagrad',
                                 'rmsprop',
                                 'adamax',
                                 'asgd',
                                 'adam',
                                 'sgd'],
                        default='adam')
    parser.add_argument('--include_punctuation',
                        action='store_false',
                        dest='ignore_punctuation')
    parser.add_argument('--sorted_batches',
                        action='store_true',
                        dest='sorted_batches')
    parser.add_argument('--mask_invalid_grads',
                        action='store_true',
                        dest='mask_invalid_grads')
    parser.add_argument('--invalid_set_to',
                        type=float,
                        default=-2) # -2 = log(0.01)
    parser.add_argument('--log_values',
                        action='store_true',
                        dest='log_values')
    parser.add_argument('--weighted_mse',
                        action='store_true',
                        dest='weighted_mse')
    parser.add_argument('--shuffle_sentences',
                        action='store_true',
                        dest='shuffle_sentences')
    parser.add_argument('--seed',
                        type=int,
                        default=1234)
    parser.add_argument('--split_dir',
                        type=str,
                        default='libritts/with_pos')
    parser.add_argument('--exp_name',
                        dest='exp_name',
                        type=str,
                        default='test_experiment')
    parser.add_argument('--use_pos',
                        dest='use_pos',
                        action=BooleanOptionalAction)
    parser.add_argument('--extract_pos',
                        action=BooleanOptionalAction)
    parser.add_argument('--freeze',
                        action=BooleanOptionalAction)
    parser.add_argument('--train',
                        action=BooleanOptionalAction)
    parser.add_argument('--test',
                        action=BooleanOptionalAction)
    parser.add_argument('--predict',
                        action=BooleanOptionalAction)
    parser.add_argument('--layer_weights',
                        action=BooleanOptionalAction)
    parser.add_argument('-cp',
                        '--checkpoint',
                        type=str,
                        nargs='+',
                        dest='checkpoint',
                        default='[]')
    parser.add_argument('-cpdir',
                        '--checkpoint_dir',
                        type=str,
                        dest='checkpoint_dir',
                        default='./checkpoints')
    parser.add_argument('-ls',
                        '--label_smoothing',
                        type=float,
                        dest='label_smoothing',
                        default=0.0)
    # parse arguments
    return parser.parse_args()


def make_dirs(name):
    try:
        os.makedirs(name)
    except OSError as ex:
        if ex.errno == errno.EEXIST and os.path.isdir(name):
            # ignore existing directory
            pass
        else:
            # a different error happened
            raise


def weighted_mse_loss(input,target):
    tgt_device = target.device
    BUFFER = torch.Tensor([3.0]).to(tgt_device)
    SOFT_MAX_BOUND = torch.Tensor([6.0]).to(tgt_device) + BUFFER
    weights = (torch.min(target + BUFFER, SOFT_MAX_BOUND) / SOFT_MAX_BOUND)
    weights = weights / torch.sum(weights)
    weights = weights.cuda()
    sq_err = (input-target)**2
    weighted_err = sq_err * weights.expand_as(target)
    loss = weighted_err.mean()
    return loss


def main():

    # parse arguments
    config = config_parser()

    # set seeds
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    random.seed(config.seed)

    if config.gpu is not None:
        torch.cuda.set_device(config.gpu)
        device = torch.device('cuda:{}'.format(config.gpu))
        torch.cuda.manual_seed(config.seed)
        print("\nTraining on GPU[{}] (torch.device({})).".format(config.gpu, device))
    else:
        device = torch.device('cpu')
        print("GPU not available so training on CPU (torch.device({})).".format(device))
        device = 'cpu'

    # optimizer
    if config.optimizer == 'adadelta':
        optim_algorithm = optim.Adadelta
    elif config.optimizer == 'adagrad':
        optim_algorithm = optim.Adagrad
    elif config.optimizer == 'adam':
        optim_algorithm = optim.Adam
    elif config.optimizer == 'adamax':
        optim_algorithm = optim.Adamax
    elif config.optimizer == 'asgd':
        optim_algorithm = optim.ASGD
    elif config.optimizer == 'rmsprop':
        optim_algorithm = optim.RMSprop
    elif config.optimizer == 'rprop':
        optim_algorithm = optim.Rprop
    elif config.optimizer == 'sgd':
        optim_algorithm = optim.SGD
    else:
        raise Exception('Unknown optimization optimizer: "%s"' % config.optimizer)

    # load data from file
    if config.extract_pos:
        add_pos_to_dataset(config)
    splits, tag2index, index_to_tag, vocab, pos2index, index2pos = prosody_dataset.load_dataset(config)

    # model definition
    if config.model == "BertUncased" or config.model == "BertCased":
        model = Bert(device, config, labels=len(tag2index), freeze=config.freeze) if not config.use_pos else Bert(device, config, labels=len(pos2index), freeze=config.freeze)
    elif config.model == "BertLSTM":
        model = BertLSTM(device, config, labels=len(tag2index))
    elif config.model == "LSTM" or config.model == "BiLSTM":
        model = LSTM(device, config, vocab_size=len(vocab), labels=len(tag2index))
    elif config.model == "BertRegression":
        model = BertRegression(device, config)
        config.ignore_punctuation = True
    elif config.model == "LSTMRegression":
        model = LSTMRegression(device, config, vocab_size=len(vocab))
        config.ignore_punctuation = True
    elif config.model == "WordMajority":
        model = WordMajority(device, config, index_to_tag)
    elif config.model == "ClassEncodings":
        model = ClassEncodings(device, config, index_to_tag, tag2index)
    elif config.model == "BertAllLayers":
        model = BertAllLayers(device, config, labels=len(tag2index))
    else:
        raise NotImplementedError("Model option not supported.")

    model.to(device)

    # word embeddings
    if config.model == 'LSTM' or config.model == 'BiLSTM':
        weights, word_to_embid = load_embeddings(config, vocab)
        model.word_embedding.weight.data = torch.Tensor(weights).to(device)
    else:
        word_to_embid = None

    # load torch datasets
    print(f'\nLoading datasets...')
    train_dataset = Dataset(splits["train"], tag2index, pos2index, config, word_to_embid, desc="train")
    eval_dataset = Dataset(splits["dev"], tag2index, pos2index, config, word_to_embid, desc="dev")
    test_dataset = Dataset(splits["test"], tag2index, pos2index, config, word_to_embid, desc="test")

    train_iter = data.DataLoader(dataset=train_dataset,
                                 batch_size=config.batch_size,
                                 shuffle=not(config.sorted_batches), # will manually shuffle if sorted_batches desired
                                 num_workers=1,
                                 collate_fn=prosody_dataset.pad)
    dev_iter = data.DataLoader(dataset=eval_dataset,
                               batch_size=config.batch_size,
                               shuffle=False,
                               num_workers=1,
                               collate_fn=prosody_dataset.pad)
    test_iter = data.DataLoader(dataset=test_dataset,
                                batch_size=config.batch_size,
                                shuffle=False,
                                num_workers=1,
                                collate_fn=prosody_dataset.pad)

    if config.model in ["WordMajority"]:
        optimizer = None
    else:
        # add custom learning rate for individual layers in the model
        layer_names = [name for (name,param) in model.named_parameters()]
        weights_lr = 1e-2
        target_layer = 'weight.weight_layers'
        layer_names = list(set(layer_names)-set([target_layer]))
        params = [{'params': [value for name, value in model.named_parameters() if name in layer_names], 'lr':config.learning_rate},\
                  {'params': [value for name, value in model.named_parameters() if name not in layer_names], 'lr':weights_lr}]
        # init optimizer
        optimizer = optim_algorithm(params,
                                     lr=config.learning_rate,
                                     weight_decay=config.weight_decay)
        # optimizer = optim_algorithm(model.parameters(),
        #                             lr=config.learning_rate,
        #                             weight_decay=config.weight_decay)

    if config.model in ['BertRegression', 'LSTMRegression']:
        if config.weighted_mse:
            criterion = weighted_mse_loss
        else:
            criterion = nn.MSELoss()
    elif config.model == 'ClassEncodings':
        criterion = nn.BCELoss()
    else:
        # criterion = nn.CrossEntropyLoss(ignore_index=0)
        criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0)

    params = sum([p.numel() for p in model.parameters()])
    print('Parameters: {}'.format(params))

    config.cells = config.layers

    if config.model == 'BiLSTM':
        config.cells *= 2

    if config.model == 'WordMajority': # 1 pass over the dataset is enough to collect stats
        config.epochs = 1

    # select mode [train / validate / predict]
    if config.train:
        # init training args
        print('\nTraining started...\n')
        best_dev_acc = 0
        best_dev_epoch = 0
        training_start_time = time.time()

        if config.model in ['BertRegression', 'LSTMRegression']:
            for epoch in range(config.epochs):
                print("Epoch: {}".format(epoch + 1))
                train_cont(model, train_iter, optimizer, criterion, device, config)
                valid_cont(model, dev_iter, criterion, index_to_tag, device, config, best_dev_acc, best_dev_epoch, epoch + 1)
            test_cont(model, test_iter, criterion, index_to_tag, device, config)

        else:
            for epoch in range(config.epochs):
                print("Epoch: {}".format(epoch+1))
                train(model, train_iter, optimizer, criterion, device, config)
                valid(model, dev_iter, criterion, index_to_tag if not config.use_pos else index2pos, device, config, best_dev_acc, best_dev_epoch, epoch+1)
            test(model, test_iter, criterion, index_to_tag if not config.use_pos else index2pos, device, config)

        # print training time
        m, s = divmod((time.time() - training_start_time), 60)
        print(f'Training finished! Time elapsed: {m:.1f} minutes and {s:.1f} seconds')
    elif config.test:
        # load model(s) from checkpoint(s)
        if len(config.checkpoint) > 1:
            # fuse multiple models
            print(f'\nFusing multiple models from checkpoints...')
            ckpts = []
            for i, ckpt in enumerate(config.checkpoint):
                if not Path(ckpt).exists():
                    raise Exception('Checkpoint not available. Please specify a correct path/file to checkpoint.')
                print(f'\nLoading state dictionary from checkpoint {ckpt}...')
                if i == 0:
                    global_model = torch.load(ckpt, map_location=device).state_dict()
                else:
                    ckpts.append(torch.load(ckpt, map_location=device).state_dict())
            # average all parameters
            for key in global_model:
                sub_models = torch.stack([k[key] for k in ckpts]).type(torch.DoubleTensor).to(device)
                global_model[key] = torch.mean(torch.vstack((global_model[key].unsqueeze(0), sub_models)), dim=0)
            model.load_state_dict(global_model)
        else:
            # load single model
            if not Path(config.checkpoint[0]).exists():
                raise Exception('Checkpoint not available. Please specify a correct path/file to checkpoint.')
            print(f'\nLoading state dictionary from checkpoint {config.checkpoint[0]}...')
            model.load_state_dict(torch.load(config.checkpoint[0], map_location=device).state_dict())
        model.to(device)
        # run model on test data
        test(model, test_iter, criterion, index_to_tag, device, config)
    elif config.predict:
        # predict endpoint not implemented yet
        pass
    else:
        raise Exception('Unsupported argument. Please choose between [train/test/predict]')


# --------------- FUNCTIONS FOR DISCRETE MODELS --------------------

def train(model, iterator, optimizer, criterion, device, config):
    model.train()
    for i, batch in enumerate(iterator):
        words, x, is_main_piece, tags, y, seqlens, _, pids, pos, _ = batch

        if config.model == 'WordMajority':
            model.collect_stats(x, y)
            continue

        optimizer.zero_grad()
        x = x.to(device)
        y = pids.to(device) if config.use_pos else y.to(device)
        #import pdb;pdb.set_trace()

        logits, y, _ = model(x, y) # logits: (N, T, VOCAB), y: (N, T)

        if config.model == 'ClassEncodings':
            logits = logits.view(-1, logits.shape[-1])  # (N*T, VOCAB)
            y = y.view(-1, y.shape[-1])  # also (N*T, VOCAB)
            loss = criterion(logits.to(device), y.to(device))
        else:
            logits = logits.view(-1, logits.shape[-1]) # (N*T, VOCAB)
            y = y.view(-1)  # (N*T,)
            loss = criterion(logits.to(device), y.to(device))
            # import pdb;pdb.set_trace()
            # loss = criterion(logits.to(device).permute(0,2,1), y.to(device))

        loss.backward()
        optimizer.step()

        if i % config.log_every == 0 or i+1 == len(iterator):
            print("Training step: {}/{}, loss: {:<.4f}".format(i+1, len(iterator), loss.item()))

    if config.model == 'WordMajority':
        model.save_stats()


def valid(model, iterator, criterion, index_to_tag, device, config, best_dev_acc, best_dev_epoch, epoch):
    if config.model == 'WordMajority':
        return

    model.eval()
    dev_losses = []
    Words, POS, Is_main_piece, Tags, Y, Y_hat = [], [], [], [], [], []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            words, x, is_main_piece, tags, y, seqlens, _, pids, pos, _ = batch
            x = x.to(device)
            y = pids.to(device) if config.use_pos else y.to(device)

            logits, labels, y_hat = model(x, y)  # y_hat: (N, T)

            if config.model == 'ClassEncodings':
                logits = logits.view(-1, logits.shape[-1])  # (N*T, VOCAB)
                labels = labels.view(-1, labels.shape[-1])  # also (N*T, VOCAB)
                loss = criterion(logits.to(device), labels.to(device))
            else:
                logits = logits.view(-1, logits.shape[-1])  # (N*T, VOCAB)
                labels = labels.view(-1)  # (N*T,)
                loss = criterion(logits.to(device), labels.to(device))

            dev_losses.append(loss.item())

            Words.extend(words)
            POS.extend(pos)
            Is_main_piece.extend(is_main_piece)
            Tags.extend(tags)
            Y.extend(y.cpu().numpy().tolist())
            Y_hat.extend(y_hat.cpu().numpy().tolist())

    true = []
    predictions = []
    # gen_pack = (x,y,z,k for x,y,z,k in zip(Words, Is_main_piece, POS if config.use_pos else Tags, Y_hat)
    for words, is_main_piece, tags, y_hat in zip(Words, Is_main_piece, POS if config.use_pos else Tags, Y_hat):
        y_hat = [hat for head, hat in zip(is_main_piece, y_hat) if head == 1]
        preds = [index_to_tag[hat] for hat in y_hat]

        if config.model != 'LSTM' and config.model != 'BiLSTM':
            tagslice = tags.split()[1:-1]
            predsslice = preds[1:-1]
            assert len(preds) == len(words.split()) == len(tags.split())
        else:
            tagslice = tags.split()
            predsslice = preds
        for t, p in zip(tagslice, predsslice):
            if config.ignore_punctuation:
                if t != 'NA':
                    true.append(t)
                    predictions.append(p)
            else:
                true.append(t)
                predictions.append(p)

    # calc metric
    y_true = np.array(true)
    y_pred = np.array(predictions)
    acc = 100. * (y_true == y_pred).astype(np.int32).sum() / len(y_true)

    if acc > best_dev_acc:
        best_dev_acc = acc
        best_dev_epoch = epoch
        dev_snapshot_path = 'best_model_{}_devacc_{}_epoch_{}.pt'.format(config.model, round(best_dev_acc, 2), best_dev_epoch)

        # save model, delete previous snapshot
        torch.save(model, Path('.').joinpath(config.checkpoint_dir, dev_snapshot_path))
        for f in glob.glob(Path('.').joinpath(config.checkpoint_dir, 'best_model_*').as_posix()):
            if f != dev_snapshot_path:
                os.remove(f)

    print('Validation accuracy: {:<5.2f}%, Validation loss: {:<.4f}\n'.format(round(acc, 2), np.mean(dev_losses)))


def test(model, iterator, criterion, index_to_tag, device, config):
    print('Calculating test accuracy and printing predictions to file {}'.format(config.save_path))
    print("Output file structure: <word>\t <tag>\t <prediction>\n")

    model.eval()
    test_losses = []

    Words, POS, Is_main_piece, Tags, Y, Y_hat = [], [], [], [], [], []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            words, x, is_main_piece, tags, y, seqlens, _, pids, pos, _ = batch
            x = x.to(device)
            y = y.to(device)
            logits, labels, y_hat = model(x, y)  # y_hat: (N, T)

            if config.model == 'ClassEncodings':
                logits = logits.view(-1, logits.shape[-1])  # (N*T, VOCAB)
                labels = labels.view(-1, labels.shape[-1])  # also (N*T, VOCAB)
                loss = criterion(logits.to(device), labels.to(device))
            else:
                logits = logits.view(-1, logits.shape[-1])  # (N*T, VOCAB)
                labels = labels.view(-1)  # (N*T,)
                loss = criterion(logits, labels)

            test_losses.append(loss.item())

            Words.extend(words)
            POS.extend(pos)
            Is_main_piece.extend(is_main_piece)
            Tags.extend(tags)
            Y.extend(y.cpu().numpy().tolist())
            Y_hat.extend(y_hat.cpu().numpy().tolist())

    true = []
    predictions = []
    # gets results and save
    with open(config.save_path, 'w') as results:
        for words, is_main_piece, tags, y_hat in zip(Words, Is_main_piece, POS if config.use_pos else Tags, Y_hat):
            y_hat = [hat for head, hat in zip(is_main_piece, y_hat) if head == 1]
            preds = [index_to_tag[hat] for hat in y_hat]
            if config.model != 'LSTM' and config.model != 'BiLSTM':
                tagslice = tags.split()[1:-1]
                predsslice = preds[1:-1]
                wordslice = words.split()[1:-1]
                assert len(preds) == len(words.split()) == len(tags.split())
            else:
                tagslice = tags.split()
                predsslice = preds
                wordslice = words.split()
            for w, t, p in zip(wordslice, tagslice, predsslice):
                results.write("{}\t{}\t{}\n".format(w, t, p))
                if config.ignore_punctuation:
                    if t != 'NA':
                        true.append(t)
                        predictions.append(p)
                else:
                    true.append(t)
                    predictions.append(p)
            results.write("\n")

    # calc metric
    y_true = np.array(true)
    y_pred = np.array(predictions)

    acc = 100. * (y_true == y_pred).astype(np.int32).sum() / len(y_true)
    print(f"Test accuracy: {round(acc, 2):<5.2f}%, Test loss: {np.mean(test_losses):<.4f} after {config.epochs} epochs.\n")
    final_snapshot_path = f"model_{config.model}_{'frozen' if config.freeze else 'finetuned'}_testacc_{round(acc, 2)}_epoch_{config.epochs}_batch_{config.batch_size}_{config.exp_name}_{Path(config.split_dir).name}.pt"

    # print classification report
    # print(classification_report(y_true, y_pred, target_names=['no_accent','accent']))
    print(classification_report(y_true, y_pred))

    # save final model checkpoint
    torch.save(model, Path('.').joinpath(config.checkpoint_dir, final_snapshot_path))


def predict(model, iterator, criterion, index_to_tag, device, config):
    # placeholder for predict method
    pass


# ---------------- FUNCTIONS FOR CONTINUOUS MODELS ------------------
''' These are used only the BertRegression and LSTMRegression models for now '''

def train_cont(model, iterator, optimizer, criterion, device, config):

    model.train()
    for i, batch in enumerate(iterator):
        words, x, is_main_piece, tags, y, seqlens, values, invalid_set_to = batch

        optimizer.zero_grad()
        x = x.to(device)
        values = values.to(device)

        predictions, true = model(x, values)
        loss = criterion(predictions.to(device), true.float().to(device))
        loss.backward()
        optimizer.step()

        if i % config.log_every == 0 or i+1 == len(iterator):
            print("Training step: {}/{}, loss: {:<.4f}".format(i+1, len(iterator), loss.item()))


def valid_cont(model, iterator, criterion, index_to_tag, device, config, best_dev_acc, best_dev_epoch, epoch):
    model.eval()
    dev_losses = []
    Words, Is_main_piece, Tags, Y, Predictions, Values = [], [], [], [], [], []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            words, x, is_main_piece, tags, y, seqlens, values, invalid_set_to = batch
            x = x.to(device)
            values = values.to(device)

            predictions, true = model(x, values)
            loss = criterion(predictions.to(device), true.float().to(device))
            dev_losses.append(loss.item())

            Words.extend(words)
            Is_main_piece.extend(is_main_piece)
            Tags.extend(tags)
            Y.extend(y.cpu().numpy().tolist())
            Predictions.extend(predictions.cpu().numpy().tolist())
            Values.extend(values.cpu().numpy().tolist())

    print('Validation loss: {:<.4f}\n'.format(np.mean(dev_losses)))


def test_cont(model, iterator, criterion, index_to_tag, device, config):
    print('Calculating test accuracy and printing predictions to file {}'.format(config.save_path))
    print("Output file structure: <word>\t <tag>\t <prediction>\n")

    model.eval()
    test_losses = []

    Words, Is_main_piece, Tags, Y, Predictions, Values = [], [], [], [], [], []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            words, x, is_main_piece, tags, y, seqlens, values, invalid_set_to = batch
            x = x.to(device)
            values = values.to(device)

            predictions, true = model(x, values)
            loss = criterion(predictions.to(device), true.float().to(device))
            test_losses.append(loss.item())

            Words.extend(words)
            Is_main_piece.extend(is_main_piece)
            Tags.extend(tags)
            Y.extend(y.cpu().numpy().tolist())
            Predictions.extend(predictions.cpu().numpy().tolist())
            Values.extend(values.cpu().numpy().tolist())

    true = []
    preds_to_eval = []
    # gets results and save
    with open(config.save_path, 'w') as results:
        for words, is_main_piece, tags, preds, values in zip(Words, Is_main_piece, Tags, Predictions, Values):
            valid_preds = [p for head, p in zip(is_main_piece, preds) if head == 1]

            predsslice = valid_preds[1:-1]
            valuesslice = values[1:-1]
            wordslice = words.split()[1:-1]

            for w, v, p in zip(wordslice, valuesslice, predsslice):
                results.write("{}\t{}\t{}\n".format(w, v, p))
                if v != invalid_set_to:
                    true.append(v)
                    preds_to_eval.append(p)
            results.write("\n")
    # calc metric
    y_true = np.array(true)
    y_pred = np.array(preds_to_eval)

    print('Test loss: {:<.4f}\n'.format(np.mean(test_losses)))
    # Correlation is calculated afterwards with a separate script.


if __name__ == "__main__":
    main()
