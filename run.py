import re
import time
from utils import get_blosum62
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from Bio import Entrez
from Bio import SeqIO
from numba import jit


def hashnchain(seq, k, nt_flag):
    ''' hash the k-mers and chain the next instance

    :param seq: nucleotide or amino acid sequence
    :param k: k-mer size
    :param nt_flag: if the sequence is  #TODO Remove nt_flag
    :return: Returns the hashed and chain lookup table
    '''
    if nt_flag:
        val_dict = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        num_vals = 4
    else:
        num_vals = 20
        val_dict = {
            "A": 0, "C": 1, "D": 2,
            "E": 3, "F": 4, "G": 5,
            "H": 6, "I": 7, "K": 8,
            "L": 9, "M": 10, "N": 11,
            "P": 12, "Q": 13, "R": 14,
            "S": 15, "T": 16, "V": 17,
            "W": 18, "Y": 19,
        }

    tuples = len(seq) - k + 1
    a = np.full(num_vals ** k, np.nan, dtype=object)
    b = np.full(num_vals ** k, np.nan, dtype=object)
    for i in range(tuples):
        c = 0
        s = ''
        for j in range(k):
            c += (val_dict[seq[i + j]] * (num_vals ** ((k - 1) - j)))
            s += seq[i + j]
        if np.isnan(a[c]).any():
            a[c] = [i]
            b[c] = s
        else:
            a[c].append(i)
    return a, b


def FASTA(q_seq, t_seq, k=6, diag_min_score=3, allowed_gap=1, gap_pen=1):
    if len(q_seq) > len(t_seq):
        temp = q_seq
        q_seq = t_seq
        t_seq = temp
        print('Query/Target swap')

    q_hash_list, q_hash_value = hashnchain(q_seq, k, True)
    t_hash_list, t_hash_value = hashnchain(t_seq, k, True)
    num_diags = len(q_seq) + len(t_seq) - (2 * k) + 1

    diags = []
    diag_qchars = []
    diag_tchars = []
    count = 1
    for i in range(num_diags):
        if i < len(q_seq) - k:
            diags.append(np.zeros((k + i), dtype=int))
            diag_qchars.append(np.zeros((k + i),dtype=int))
            diag_tchars.append(np.zeros((k + i),dtype=int))
        elif i >= len(t_seq) - k + 1:
            diags.append(np.zeros((len(q_seq) - count), dtype=int))
            diag_qchars.append(np.zeros((len(q_seq) - count), dtype=int))
            diag_tchars.append(np.zeros((len(q_seq) - count), dtype=int))
            count += 1
        else:
            diags.append(np.zeros((len(q_seq)), dtype=int))
            diag_qchars.append(np.zeros((len(q_seq)), dtype=int))
            diag_tchars.append(np.zeros((len(q_seq)), dtype=int))

    for i, qk in enumerate(q_hash_list):
        if not np.isnan(qk).any() and not np.isnan(t_hash_list[i]).any():
            for i_qk in qk:
                for i_tk in t_hash_list[i]:
                    i_diag = len(q_seq) - k - i_qk + i_tk
                    diag_idx = 0
                    if i_diag < len(q_seq) - k:
                        diag_idx = i_tk
                    elif i_diag == len(q_seq):
                        diag_idx = i_qk
                    elif i_diag >= len(q_seq) - len(t_seq):
                        diag_idx = i_qk

                    diags[i_diag][diag_idx:(diag_idx + k)] = 1
                    for ks in range(k):
                        diag_qchars[i_diag][diag_idx + ks] = i_qk + ks
                        diag_tchars[i_diag][diag_idx + ks] = i_tk + ks
    hotspots = []
    gaps = []
    scores = []
    hotspot_qseqs = []
    hotspot_tseqs = []


    for i, diag in enumerate(diags):
        hotspot = []
        start = np.nan
        end = np.nan
        h_gap = 0
        gap = []
        score = []
        hotspot_qseq = []
        hotspot_tseq = []
        if diag.sum() > diag_min_score:
            for j, el in enumerate(diag[:]):
                if el == 1:
                    if np.isnan(start):
                        start = j
                        end = j
                    else:
                        end = j
                else:
                    if not np.isnan(start) and allowed_gap != 0 and diag[j:j+(allowed_gap+1)].sum() > 0:
                        end = j
                        h_gap += len(np.nonzero(diag[j:j+(allowed_gap+1)]))
                    elif not np.isnan(start) and not np.isnan(end):
                        hotspot.append([start, end])
                        gap.append([h_gap])
                        score.append([(end - start + 1) - (h_gap * gap_pen)])
                        hotspot_qseq.append([q_seq[diag_qchars[i][start]:diag_qchars[i][end]+1]])
                        hotspot_tseq.append([t_seq[diag_tchars[i][start]:diag_tchars[i][end]+1]])
                        start = np.nan
                        end = np.nan
                        h_gap = 0
                    else:
                        continue
            if not np.isnan(start) and not np.isnan(end):
                hotspot.append([start, end])
                gap.append([h_gap])
                score.append([(end - start + 1) - (h_gap * gap_pen)])
                hotspot_qseq.append([q_seq[diag_qchars[i][start]:diag_qchars[i][end]+1]])
                hotspot_tseq.append([t_seq[diag_tchars[i][start]:diag_tchars[i][end]+1]])

        hotspots.append(hotspot)
        gaps.append(gap)
        scores.append(score)
        hotspot_qseqs.append(hotspot_qseq)
        hotspot_tseqs.append(hotspot_tseq)

    for i, diag in enumerate(diags):
        score = diag.sum()
        if score > diag_min_score:
            print(f'Diag {i: 4d} with score {score: 4d}')
            print(scores[i])
    print('End')

def database_search(query, database_list):
    Entrez.email = 'chris.f.angelini@gmail.com'
    handle = Entrez.efetch(db="nucleotide", id=query, rettype="gb", retmode="text")
    q_record = SeqIO.read(handle, "genbank")
    handle.close()
    print(f'{q_record.id} \n {q_record.description}')
    query_seq = str(q_record.seq)

    initn = []
    for target in database_list:
        handle = Entrez.efetch(db="nucleotide", id=target, rettype="gb", retmode="text")
        record = SeqIO.read(handle, "genbank")
        handle.close()
        print(f'Align {q_record.id} and {record.id}')
        print(f'Length: {len(q_record)} | Seq: {repr(q_record.seq)}')
        print(f'Length: {len(record)} | Seq: {repr(record.seq)}')
        target_seq = str(record.seq)

        clock0 = time.time()
        FASTA(query_seq, target_seq, diag_min_score=100, k=3, allowed_gap=40, gap_pen=1)
        clock1 = time.time()
        print(f' {(clock1 - clock0) * 1000: 2.4f} ms')
    return initn

if __name__ == '__main__':

    query_acc = "AY707088"
    database_accs = ["X79493"]

    database_search(query_acc, database_accs)

    #target = 'ccatcggcatcg'.upper()
    #query = 'gcataggc'.upper()
    #FASTA(query, target, diag_min_score=3, k=3, allowed_gap=1, gap_pen=1)
    #clock1 = time.time()



