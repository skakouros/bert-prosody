from pathlib import Path
import pandas as pd
from ast import literal_eval
from typing import List, Tuple, Iterable
import numpy as np
from itertools import islice
import re


def chunk(it: Iterable = None, size: int = None) -> Iterable:
    it = iter(it)
    return list(iter(lambda: tuple(islice(it, size)), ()))


def split_punctuation(utterance: List[str], prominence: List[str], duration: List[str], add_eos_mark: bool = False) \
        -> Tuple[List[str], List[str], List[str]]:
    # definitions
    puncs = [',', '.']
    utts, proms, durs = [], [], []
    punc_checker = lambda w: [p for p in puncs if p in w]
    prom_offset = 0  # this is the label offset for prominence tags
    pattern = r"[\([{})\]]"  # regexp pattern to remove any type of brackets in the input

    # iterate
    for w, p, d in zip(utterance, prominence, duration):
        w = re.sub(pattern, '', w)
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


def convert_burnc(filename: str = None, new_suffix: str = '.txt', split: bool = True, seed: int = 444,
                  chunk_size: int = 2) -> None:
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
    split_names = ['train', 'dev', 'test']
    splits = {y: x.reset_index() for x, y in
              zip(np.split(df.sample(frac=1, random_state=seed), [int(.85 * len(df)), int(.95 * len(df))]),
                  split_names)} if split else {}
    where = lambda wlist, idx: [jj for jj, w in enumerate(wlist) if idx == w]
    wslice = lambda wlist, idxs: [wlist[0 if jj == 0 else idxs[jj - 1] + 1:idx + 1] for jj, idx in enumerate(idxs)]

    # process file(s)
    output_files = [Path(filename).with_suffix(new_suffix)] if not split else [f(filename, x) for x in split_names]
    for output_file in output_files:
        with open(output_file, 'w') as fn:
            cdf = df if not split else splits[Path(output_file).stem]
            for ii in cdf.index:
                # read row
                fname, speaker = Path(cdf.loc[ii, 'textgrid_path']).name, cdf.loc[ii, 'speaker']
                words, prominence, duration, pos = literal_eval(cdf.loc[ii, 'word']), literal_eval(
                    cdf.loc[ii, 'prominence']), literal_eval(cdf.loc[ii, 'duration']), literal_eval(cdf.loc[ii, 'pos'])
                # update with punctuations and clear any unwanted punctuation marks
                words, prominence, duration = split_punctuation(words, prominence, duration)
                if chunk_size is None:
                    # write to file
                    header_writer(fn, fname)
                    rows_writer(fn, words, prominence, duration)
                else:
                    # determine if row has more than one sentences
                    idxs = where(words, '.')
                    if len(idxs) > 1 and len(idxs) > chunk_size:
                        words_li, prominence_li, duration_li, pos_li = wslice(words, idxs), wslice(prominence, idxs), \
                                                                       wslice(duration, idxs), wslice(pos, idxs)
                        words_li, prominence_li, duration_li, pos_li = chunk(words_li, chunk_size), \
                                                                       chunk(prominence_li, chunk_size), \
                                                                       chunk(duration_li, chunk_size), \
                                                                       chunk(pos_li, chunk_size)
                        for i, (words, prominence, duration) in enumerate(
                                zip(words_li, prominence_li, duration_li)):
                            # write to file
                            header_writer(fn, fname + f'.S{i}')
                            for w, p, d in zip(words, prominence, duration):
                                rows_writer(fn, w, p, d)
                    else:
                        # write to file
                        header_writer(fn, fname)
                        rows_writer(fn, words, prominence, duration)


def convert_swbdnxt(filename: str = None, new_suffix: str = '.txt', split: bool = True, split_speakers: bool = False,
                        seed: int = 444, chunk_size: int = 4) -> None:
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
    split_names = ['train', 'dev', 'test']
    splits = {y: x.reset_index() for x, y in
              zip(np.split(df.sample(frac=1, random_state=seed), [int(.85 * len(df)), int(.95 * len(df))]),
                  split_names)} if split else {}

    # process file
    output_files = [Path(filename).with_suffix(new_suffix)] if not split else [f(filename, x) for x in split_names]
    for output_file in output_files:
        with open(output_file, 'w') as fn:
            cdf = df if not split else splits[Path(output_file).stem]
            for ii in cdf.index:
                if split_speakers:
                    for sid in ['SA', 'SB']:
                        all_sentences, speaker = literal_eval(cdf.loc[ii, 'sentence']), cdf.loc[ii, f'{sid}_speaker_ID']
                        fname = Path(cdf.loc[ii, 'textgrid_path']).name
                        if chunk_size is None:
                            words, prominence, duration = literal_eval(cdf.loc[ii, f'{sid}_word']), literal_eval(
                                cdf.loc[ii, f'{sid}_accent']), literal_eval(cdf.loc[ii, f'{sid}_duration'])
                            # update with punctuations
                            words, prominence, duration = split_punctuation(words, prominence, duration)
                            # write to file
                            header_writer(fn, fname + f'.{sid}')
                            rows_writer(fn, words, prominence, duration)
                        else:
                            # TODO: The chunking code below needs correction; currently it returns chunks of N words
                            #  and not N sentences (this is because annotation is missing the sentence division and
                            #  instead the entire speaker transcription has been stored as a single list without full
                            #  stops).
                            words_li, prominence_li, duration_li = \
                                chunk(literal_eval(cdf.loc[ii, f'{sid}_word']), chunk_size), \
                                chunk(literal_eval(cdf.loc[ii, f'{sid}_accent']), chunk_size), \
                                chunk(literal_eval(cdf.loc[ii, f'{sid}_duration']), chunk_size)
                            for i, (words, prominence, duration) in enumerate(
                                    zip(words_li, prominence_li, duration_li)):
                                # update with punctuations
                                words, prominence, duration = split_punctuation(words, prominence, duration)
                                # write to file
                                header_writer(fn, fname + f'.{sid}.S{i}')
                                rows_writer(fn, words, prominence, duration)

                else:
                    sentences, speaker = literal_eval(cdf.loc[ii, 'sentence']), cdf.loc[ii, 'SA_speaker_ID']
                    fname = Path(cdf.loc[ii, 'textgrid_path']).name
                    if chunk_size is None:
                        words, prominence, duration = literal_eval(cdf.loc[ii, 'word']), literal_eval(
                            cdf.loc[ii, 'prominence']), literal_eval(cdf.loc[ii, 'duration'])
                        # update with punctuations
                        split_data = [split_punctuation(w, p, d, add_eos_mark=True) for w, p, d in
                                      zip(words, prominence, duration)]
                        words, prominence, duration = flatten(split_data, 0), flatten(split_data, 1), flatten(
                            split_data, 2)
                        # write to file
                        header_writer(fn, fname)
                        rows_writer(fn, words, prominence, duration)
                    else:
                        words_li, prominence_li, duration_li = chunk(literal_eval(cdf.loc[ii, 'word']), chunk_size), \
                                                        chunk(literal_eval(cdf.loc[ii, 'prominence']), chunk_size), \
                                                        chunk(literal_eval(cdf.loc[ii, 'duration']), chunk_size)
                        # update with punctuations and write to file
                        for i, (words, prominence, duration) in enumerate(zip(words_li, prominence_li, duration_li)):
                            split_data = [split_punctuation(w, p, d, add_eos_mark=True) for w, p, d in
                                          zip(words, prominence, duration)]
                            words, prominence, duration = flatten(split_data, 0), flatten(split_data, 1), flatten(
                                split_data, 2)
                            # write to file
                            header_writer(fn, fname + f'.S{i}')
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
