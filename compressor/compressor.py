# rle_compressed, cntrs = rle.encode(mtf.encode(BWT.encode(part_text+SHARP)))
import argparse
from typing import Tuple
import numba
import numpy as np
import string
import math
import plotly.express as px
import pandas as pd
from bitarray import bitarray, bits2bytes
from itertools import permutations
from adaptive import AdaptiveHuffman
import random
from functools import partial


HEADER_SIZE = 128
PRINTABLE = set(string.printable)
SHARP = '#'


@numba.jit()
def merge(x: np.array, SA12: np.array, SA3: np.array) -> np.array:
    "Merge the suffixes in sorted SA12 and SA3."
    ISA = np.zeros((len(x),), dtype='int')
    for i in range(len(SA12)):
        ISA[SA12[i]] = i
    SA = np.zeros((len(x),), dtype='int')
    idx = 0
    i, j = 0, 0
    while i < len(SA12) and j < len(SA3):
        if less(x, SA12[i], SA3[j], ISA):
            SA[idx] = SA12[i]
            idx += 1
            i += 1
        else:
            SA[idx] = SA3[j]
            idx += 1
            j += 1
    if i < len(SA12):
        SA[idx:len(SA)] = SA12[i:]
    elif j < len(SA3):
        SA[idx:len(SA)] = SA3[j:]
    return SA


@numba.jit()
def u_idx(i: int, m: int) -> int:
    "Map indices in u back to indices in the original string."
    if i < m:
        return 1 + 3 * i
    else:
        return 2 + 3 * (i - m - 1)


@numba.jit()
def safe_idx(x: np.array, i: int) -> int:
    "Hack to get zero if we index beyond the end."
    return 0 if i >= len(x) else x[i]


@numba.jit()
def symbcount(x: np.array, asize: int) -> np.array:
    "Count how often we see each character in the alphabet."
    counts = np.zeros((asize,), dtype="int")
    for c in x:
        counts[c] += 1
    return counts


@numba.jit()
def cumsum(counts: np.array) -> np.array:
    "Compute the cumulative sum from the character count."
    res = np.zeros((len(counts, )), dtype='int')
    acc = 0
    for i, k in enumerate(counts):
        res[i] = acc
        acc += k
    return res


@numba.jit()
def bucket_sort(x: np.array, asize: int,
                idx: np.array, offset: int = 0) -> np.array:
    "Sort indices in idx according to x[i + offset]."
    sort_symbs = np.array([safe_idx(x, i + offset) for i in idx])
    counts = symbcount(sort_symbs, asize)
    buckets = cumsum(counts)
    out = np.zeros((len(idx),), dtype='int')
    for i in idx:
        bucket = safe_idx(x, i + offset)
        out[buckets[bucket]] = i
        buckets[bucket] += 1
    return out


@numba.jit()
def radix3(x: np.array, asize: int, idx: np.array) -> np.array:
    "Sort indices in idx according to their first three letters in x."
    idx = bucket_sort(x, asize, idx, 2)
    idx = bucket_sort(x, asize, idx, 1)
    return bucket_sort(x, asize, idx)


@numba.jit()
def triplet(x: np.array, i: int) -> Tuple[int, int, int]:
    "Extract the triplet (x[i],x[i+1],x[i+2])."
    return safe_idx(x, i), safe_idx(x, i + 1), safe_idx(x, i + 2)


@numba.jit()
def collect_alphabet(x: np.array, idx: np.array) -> Tuple[np.array, int]:
    "Map the triplets starting at idx to a new alphabet."
    alpha = np.zeros((len(x),), dtype='int')
    value = 1
    last_trip = -1, -1, -1
    for i in idx:
        trip = triplet(x, i)
        if trip != last_trip:
            value += 1
            last_trip = trip
        alpha[i] = value
    return alpha, value - 1


@numba.jit()
def build_u(x: np.array, alpha: np.array) -> np.array:
    "Construct u string, using 1 as central sentinel."
    a = np.array([alpha[i] for i in range(1, len(x), 3)] +
                 [1] +
                 [alpha[i] for i in range(2, len(x), 3)])
    return a


@numba.jit()
def less(x: np.array, i: int, j: int, ISA: np.array) -> bool:
    "Check if x[i:] < x[j:] using the inverse suffix array for SA12."
    a: int = safe_idx(x, i)
    b: int = safe_idx(x, j)
    if a < b:
        return True
    if a > b:
        return False
    if i % 3 != 0 and j % 3 != 0:
        return ISA[i] < ISA[j]
    return less(x, i + 1, j + 1, ISA)


@numba.jit()
def skew_rec(x: np.array, asize: int) -> np.array:
    "skew/DC3 SA construction algorithm."

    SA12 = np.array([i for i in range(len(x)) if i % 3 != 0])

    SA12 = radix3(x, asize, SA12)
    new_alpha, new_asize = collect_alphabet(x, SA12)
    if new_asize < len(SA12):
        # Recursively sort SA12
        u = build_u(x, new_alpha)
        sa_u = skew_rec(u, new_asize + 2)
        m = len(sa_u) // 2
        SA12 = np.array([u_idx(i, m) for i in sa_u if i != m])

    if len(x) % 3 == 1:
        SA3 = np.array([len(x) - 1] + [i - 1 for i in SA12 if i % 3 == 1])
    else:
        SA3 = np.array([i - 1 for i in SA12 if i % 3 == 1])
    SA3 = bucket_sort(x, asize, SA3)
    return merge(x, SA12, SA3)


def get_suffix_array(x: str) -> np.array:
    if "$" in x:
        raise ValueError('Text should not contain $')
    str_to_int = {
        "$": 0,  # End of strig
    }
    str_to_int = str_to_int | {
        c: n+1
        for (n, c) in enumerate(sorted(list(set(x))))
    }
    return skew_rec(np.array([str_to_int[y] for y in x]), len(str_to_int))


def get_sort_canon_repr(s):
    """Returns cononical representation of sort by string s
    e.g. [3,1,0,2]"""
    sort_info = [None]*len(s)
    for new_place, (c, old_place) in enumerate(sorted([(c, i) for i, c
                                                       in enumerate(s)])):
        sort_info[old_place] = new_place
    return sort_info


def apply_permutation(s, perm):
    res = [None]*len(s)
    for old_place, new_place in enumerate(perm):
        res[new_place] = s[old_place]
    return res


def inverse_permutation(canon_repr):
    res = [None]*len(canon_repr)
    for old_place, new_place in enumerate(canon_repr):
        res[new_place] = old_place
    return res


class BWT:
    def encode(t: str):
        if SHARP not in t:
            raise ValueError(f"{SHARP}  is not found in text")
        bwt = [None]*len(t)
        sa = get_suffix_array(t)
        # print_sa(sa, t)
        for i in range(len(t)):
            bwt[i] = t[sa[i]-1]
        return ''.join(bwt)

    def decode(bwt: str):
        sigma = get_sort_canon_repr(bwt)
        inversed_sigma = inverse_permutation(sigma)
        res = [None]*len(bwt)
        i = bwt.index(SHARP)
        index_in_first_col = inversed_sigma[i]
        for j, c in enumerate(bwt):
            res[j] = bwt[index_in_first_col]
            index_in_first_col = inversed_sigma[index_in_first_col]
        return ''.join(res)


def _shift(alphabet, up, lo):
    for i in range(lo, up-1, -1):
        alphabet[i+1] = alphabet[i]
    return alphabet


class mtf:
    def get_alphabet():
        # return list('#Ban')
        return [chr(i) for i in range(ord('z')+1)]

    def update_alphabet(alphabet, ind, c):
        if ind > 1:
            _shift(alphabet, 1, ind-1)
            alphabet[1] = c
        if ind == 1:
            alphabet[1] = alphabet[0]
            alphabet[0] = c

    def encode(t: str):
        alphabet = mtf.get_alphabet()
        diff = set(t)-set(alphabet)
        if diff:
            raise ValueError(
                f'Found chars in text that are not presented in alphabet: {diff}')
        res = []
        for c in t:
            ind = alphabet.index(c)
            res.append(ind)
            mtf.update_alphabet(alphabet, ind, c)
        # print(f"mtf res max & min: {max(res), min(res)}")
        return res

    def decode(encoded):  # list of indecies
        alphabet = mtf.get_alphabet()
        res = []
        diff = set(encoded) - set(range(len(alphabet)))
        if diff:
            raise ValueError(f"Found wrong indecies in encoded by mtf: {diff}")
        for ind in encoded:
            c = alphabet[ind]
            res.append(c)
            mtf.update_alphabet(alphabet, ind, c)
        return ''.join(res)


class rle:
    def encode(ar):
        res = []
        cntrs = []
        prev_is_zero = False
        for x in ar:
            if x != 0:
                res.append(x)
                prev_is_zero = False
                continue
            if prev_is_zero:
                cntrs[-1] += 1
            else:
                res.append(0)
                cntrs.append(1)
            prev_is_zero = True
        # print(f"max in cntrs = {max(cntrs)}")
        return res, cntrs

    def decode(rle_encoded, cntrs):
        res = []
        for x in rle_encoded:
            if x != 0:
                res.append(x)
                continue
            res.extend([0]*cntrs.pop(0))
        return res


class Huffman:
    def encode(l: list):
        ada_huff = AdaptiveHuffman(bytes(l))
        return ada_huff.encode()

    def decode(code: bitarray):
        add_huff_decoder = AdaptiveHuffman(code)
        return add_huff_decoder.decode()


def gamma_code(number_bits):
    number = int(''.join(map(str, number_bits)), 2)
    # number += 1  # кодируем число на 1 больше чтобы могли подавать нули
    i = math.floor(math.log(number, 2))
    return bitarray([0]*i + [int(c) for c in "{0:b}".format(number)[-(i+1):]])


def gamma_to_list_int(ar: bitarray):
    j = 0
    res = []
    zeros_cnt = 0
    while j < len(ar):
        if ar[j] == 0:
            zeros_cnt += 1
            j += 1
            continue
        number_bits = ar[j:j+zeros_cnt+1]
        number = int(number_bits.to01(), 2)
        res.append(number)
        j = j + zeros_cnt+1
        zeros_cnt = 0
    return res


def list_int_to_gamma(l):
    res = bitarray()
    for c in l:
        gamma_coded = gamma_code("{0:b}".format(c))
        res.extend(gamma_coded)
    return res


def get_header(x):
    """x - length of huffman compressed data"""
    return bitarray("{0:b}".format(x).rjust(HEADER_SIZE, '0'))


def test_gamma():
    for i in range(10):
        l = [random.randint(3, 9) for _ in range(100)]
        assert l == gamma_to_list_int(list_int_to_gamma(l))


global_cntrs = None
global_rle_compressed = None
global_mtf_encoded = None
global_huffman_compressed = None
global_huffman_len = None
global_cntrs_compressed = None


class Archiver:
    def encode(text):
        global global_cntrs
        global global_cntrs_compressed
        global global_rle_compressed
        global global_mtf_encoded
        global global_huffman_compressed
        global global_huffman_len
        if '$' in text:
            raise ValueError('Text should not contain $')
        if '#' in text:
            raise ValueError('Text should not contain #')
        unprintable = set(text)-PRINTABLE
        if unprintable:
            raise ValueError(f"Found unsupported symbols: {unprintable}")
        t = text + SHARP
        mtf_encoded = mtf.encode(BWT.encode(t))
        global_mtf_encoded = mtf_encoded
        global_mtf_encoded = mtf_encoded
        rle_compressed, cntrs = rle.encode(mtf_encoded)
        global_rle_compressed = rle_compressed
        huffman_compressed = Huffman.encode(rle_compressed)
        global_huffman_compressed = huffman_compressed
        huffman_len = len(huffman_compressed)
        global_huffman_len = huffman_len
        header = get_header(huffman_len)
        cntrs_compressed = list_int_to_gamma(cntrs)
        global_cntrs_compressed = cntrs_compressed
        global_cntrs = cntrs
        return header+huffman_compressed+cntrs_compressed

    def decode(ar: bitarray):
        header = ar[:HEADER_SIZE]
        huffman_len = int(header.to01(), 2)
        # assert huffman_len == global_huffman_len, f"{huffman_len} {global_huffman_len}"
        huffman_compressed = ar[HEADER_SIZE:HEADER_SIZE+huffman_len]
        # assert huffman_compressed == global_huffman_compressed
        cntrs_compressed = ar[HEADER_SIZE+huffman_len:]
        # assert cntrs_compressed == global_cntrs_compressed
        cntrs = gamma_to_list_int(cntrs_compressed)
        rle_compressed = Huffman.decode(huffman_compressed)
        # assert rle_compressed == global_rle_compressed
        mtf_encoded = rle.decode(rle_compressed, cntrs)
        # assert mtf_encoded == global_mtf_encoded
        return BWT.decode(mtf.decode(mtf_encoded))[:-1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", help="Input file")
    parser.add_argument("outfile", help="Otput file")
    parser.add_argument("--encode", help="encode", action="store_true")
    parser.add_argument("--decode", help="decode", action="store_true")
    args = parser.parse_args()
    # print(args.file, args.encode)
    if not args.decode and not args.encode:
        print("You should specify one of the parameters: encode or decode")
        exit()
    if args.decode and args.encode:
        print("You should specify only one of the parameters: encode or decode")
        exit()
    if args.encode:
        with open(args.infile, 'r') as file:
            text = file.read()
            bits = Archiver.encode(text)
            with open(args.outfile, 'wb') as fh:
                bits.tofile(fh)
    if args.decode:
        bits = bitarray()
        with open(args.infile, 'rb') as fh:
            bits.fromfile(fh)
            decompressed = Archiver.decode(bits)
            with open(args.outfile, 'w') as outf:
                outf.write(decompressed)
