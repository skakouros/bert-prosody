from pathlib import Path
import pandas as pd
from ast import literal_eval
from typing import List, Tuple
import numpy as np


def split_punctuation(utterance: List[str], prominence: List[str], duration: List[str], add_eos_mark: bool = False) \
        -> Tuple[List[str], List[str], List[str]]:
    # definitions
    puncs = [',', '.']
    utts, proms, durs = [], [], []
    punc_checker = lambda w: [p for p in puncs if p in w]
    prom_offset = 1  # this is the label offset for prominence tags

    # iterate
    for w, p, d in zip(utterance, prominence, duration):
        has_punc = punc_checker(w)
        if len(has_punc) > 0:
            parts = w.split(has_punc[0])
            utts.append(parts[0])
            utts.append(has_punc[0])
            proms.append(int(p)+prom_offset)
            proms.append('NA')
            durs.append(f'{float(d):.2f}')
            durs.append('NA')
        else:
            utts.append(w)
            proms.append(int(p)+prom_offset)
            durs.append(f'{float(d):.2f}')

    # add full stop mark for end-of-sentence
    if add_eos_mark:
        utts.append('.')
        proms.append('NA')
        durs.append('NA')

    return utts, proms, durs


def convert_burnc(filename: str = None, new_suffix: str = '.txt', split: bool = False, seed: int = 444) -> None:
    # init
    np.random.seed(seed)

    # read file
    try:
        df = pd.read_csv(filename)
    except OSError as e:
        print(f"{type(e)}: {e}")

    # define output handler formatters
    header_writer = lambda handle, x: handle.write(f'<file>\t{x}\n')
    row_writer = lambda handle, word, prom, dur: handle.write(f'{word}\t{prom}\tNA\t{dur}\tNA\n')
    # rows_writer = lambda handle, ws, ps, ds: [row_writer(handle, w, p, d) for w, p, d in zip(ws, ps, ds)]
    rows_writer = lambda handle, ws, ps, ds: [handle.write(f'{w}\t{p}\tNA\t{d}\tNA\n') for w, p, d in zip(ws, ps, ds)]

    # other definitions
    f = lambda file_name, split_name: Path(file_name).with_stem(split_name).with_suffix(new_suffix)
    split_names = ['train', 'validate', 'test']
    splits = {y: x.reset_index() for x, y in
              zip(np.split(df.sample(frac=1, random_state=seed), [int(.85 * len(df)), int(.95 * len(df))]),
                  split_names)} if split else {}

    # process file(s)
    output_files = [Path(filename).with_suffix(new_suffix)] if not split else [f(filename, x) for x in split_names]
    for output_file in output_files:
        with open(output_file, 'w') as fn:
            cdf = df if not split else splits[Path(output_file).stem]
            for ii in cdf.index:
                words = literal_eval(cdf.loc[ii, 'word'])
                prominence = literal_eval(cdf.loc[ii, 'prominence'])
                pos = literal_eval(cdf.loc[ii, 'pos'])
                duration = literal_eval(cdf.loc[ii, 'duration'])
                fname = Path(cdf.loc[ii, 'textgrid_path']).name
                speaker = cdf.loc[ii, 'speaker']
                # update with punctuations
                words, prominence, duration = split_punctuation(words, prominence, duration)
                # write to file
                header_writer(fn, fname)
                rows_writer(fn, words, prominence, duration)


def convert_swbdnxt(filename: str = None, new_suffix: str = '.txt', split: bool = True, seed: int = 444) -> None:
    # init
    np.random.seed(seed)

    # read file
    try:
        df = pd.read_csv(filename)
    except OSError as e:
        print(f"{type(e)}: {e}")

    # define output handler formatters
    header_writer = lambda handle, x: handle.write(f'<file>\t{x}\n')
    row_writer = lambda handle, word, prom, dur: handle.write(f'{word}\t{prom}\tNA\t{dur}\tNA\n')
    rows_writer = lambda handle, ws, ps, ds: [handle.write(f'{w}\t{p}\tNA\t{d}\tNA\n') for w, p, d in zip(ws, ps, ds)]

    # other definitions
    flatten = lambda x, idx: [y for sub_x in x for y in sub_x[idx]]
    f = lambda file_name, split_name: Path(file_name).with_stem(split_name).with_suffix(new_suffix)
    split_names = ['train', 'validate', 'test']
    splits = {y: x.reset_index() for x, y in
              zip(np.split(df.sample(frac=1, random_state=seed), [int(.85 * len(df)), int(.95 * len(df))]),
                  split_names)} if split else {}

    # process file
    output_files = [Path(filename).with_suffix(new_suffix)] if not split else [f(filename, x) for x in split_names]
    for output_file in output_files:
        with open(output_file, 'w') as fn:
            cdf = df if not split else splits[Path(output_file).stem]
            for ii in cdf.index:
                words = literal_eval(cdf.loc[ii, 'word'])
                prominence = literal_eval(cdf.loc[ii, 'prominence'])
                duration = literal_eval(cdf.loc[ii, 'duration'])
                sentences = literal_eval(cdf.loc[ii, 'sentence'])
                speaker = cdf.loc[ii, 'SA_speaker_ID']
                fname = Path(cdf.loc[ii, 'textgrid_path']).name
                # update with punctuations
                split_data = [split_punctuation(w, p, d, add_eos_mark=True) for w, p, d in
                              zip(words, prominence, duration)]
                words, prominence, duration = flatten(split_data, 0), flatten(split_data, 1), flatten(split_data, 2)
                # write to file
                header_writer(fn, fname)
                rows_writer(fn, words, prominence, duration)


def convert_dataset(filename: str = None) -> None:
    if 'burnc' in filename:
        convert_burnc(filename)
    elif 'swbdnxt' in filename:
        convert_swbdnxt(filename)


if __name__ == "__main__":
    # define vars
    swbdnxt = 'swbdnxt_full_dataset.csv'
    burnc = 'burnc_full_dataset.csv'

    # run conversion
    convert_dataset(swbdnxt)
