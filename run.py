import os
import time
from utils import get_blosum62
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 300
from Bio import Entrez
from Bio import SeqIO


def hash(seq, k, type):
    if type == 'nucleotide':
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
    for i in range(tuples):
        c = 0
        s = ''
        for j in range(k):
            c += (val_dict[seq[i + j]] * (num_vals ** ((k - 1) - j)))
            s += seq[i + j]
        if np.isnan(a[c]).any():
            a[c] = [i]
        else:
            a[c].append(i)
    return a


def FASTA(q_seq, t_seq, type, k, diag_run_thres, diag_allowed_gap, diag_gap_pen, num_top_diags, chain_gap_pen, plot):
    # Flip query and target based on length
    if len(q_seq) > len(t_seq):
        temp = q_seq
        q_seq = t_seq
        t_seq = temp
        print('Query/Target swap')

    # Create hash and chain for sequences
    q_hash_list = hash(q_seq, k, type)
    t_hash_list = hash(t_seq, k, type)
    num_diags = len(q_seq) + len(t_seq) - (2 * k) + 1

    diags = []
    diag_qchars = []
    diag_tchars = []
    count = 1

    # Setup diag list lengths
    for i in range(num_diags):
        if i < len(q_seq) - k:
            diags.append(np.zeros((k + i), dtype=int))
            diag_qchars.append(np.zeros((k + i), dtype=int))
            diag_tchars.append(np.zeros((k + i), dtype=int))
        elif i >= len(t_seq) - k + 1:
            diags.append(np.zeros((len(q_seq) - count), dtype=int))
            diag_qchars.append(np.zeros((len(q_seq) - count), dtype=int))
            diag_tchars.append(np.zeros((len(q_seq) - count), dtype=int))
            count += 1
        else:
            diags.append(np.zeros((len(q_seq)), dtype=int))
            diag_qchars.append(np.zeros((len(q_seq)), dtype=int))
            diag_tchars.append(np.zeros((len(q_seq)), dtype=int))

    # Determine Hash Matches
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

    # Find all hotspots above certain threshold, retain important information
    diags_hotspots = []
    diags_qhotspots = []
    diags_thotspots = []
    diags_gaps = []
    diags_scores = []
    diags_hotspot_qseqs = []
    diags_hotspot_tseqs = []
    for i, diag in enumerate(diags):
        hotspot = []
        qhotspot = []
        thotspot = []
        start = np.nan
        end = np.nan
        h_gap = 0
        gap = []
        score = []
        hotspot_qseq = []
        hotspot_tseq = []
        if diag.sum() > 0:
            for j, el in enumerate(diag[:]):
                if el == 1:
                    if np.isnan(start):
                        start = j
                        end = j
                    else:
                        end = j
                else:
                    if not np.isnan(start) and diag_allowed_gap != 0 and diag[j:j + (diag_allowed_gap + 1)].sum() > 0:
                        end = j
                        h_gap += 1  # len(np.nonzero(diag[j:j+(diag_allowed_gap+1)]))
                    elif not np.isnan(start) and not np.isnan(end):
                        if (end - start + 1) - (h_gap * diag_gap_pen) > diag_run_thres:
                            hotspot.append([start, end])
                            qhotspot.append([diag_qchars[i][start], diag_qchars[i][end]])
                            thotspot.append([diag_tchars[i][start], diag_tchars[i][end]])
                            gap.append(h_gap)
                            score.append((end - start + 1) - (h_gap * diag_gap_pen))
                            hotspot_qseq.append(q_seq[diag_qchars[i][start]:diag_qchars[i][end] + 1])
                            hotspot_tseq.append(t_seq[diag_tchars[i][start]:diag_tchars[i][end] + 1])
                        start = np.nan
                        end = np.nan
                        h_gap = 0
                    else:
                        continue

            if not np.isnan(start) and not np.isnan(end):
                if (end - start + 1) - (h_gap * diag_gap_pen) > diag_run_thres:
                    hotspot.append([start, end])
                    qhotspot.append([diag_qchars[i][start], diag_qchars[i][end]])
                    thotspot.append([diag_tchars[i][start], diag_tchars[i][end]])
                    gap.append(h_gap)
                    score.append((end - start + 1) - (h_gap * diag_gap_pen))
                    hotspot_qseq.append(q_seq[diag_qchars[i][start]:diag_qchars[i][end] + 1])
                    hotspot_tseq.append(t_seq[diag_tchars[i][start]:diag_tchars[i][end] + 1])

        # Start stop
        diags_hotspots.append(hotspot)
        # Start and Stop idx on Query
        diags_qhotspots.append(qhotspot)
        # Start and Stop Idx on Target
        diags_thotspots.append(thotspot)
        # Gaps
        diags_gaps.append(gap)
        # Simple score
        diags_scores.append(score)
        # Query Sequence in hotspot
        diags_hotspot_qseqs.append(hotspot_qseq)
        # Target Sequence in hotspot
        diags_hotspot_tseqs.append(hotspot_tseq)

    # Rescore with substitution matrix
    for i, diag in enumerate(diags_hotspots):
        if diag:
            for j, hotspots in enumerate(diag):
                q_hotspot = diags_hotspot_qseqs[i][j]
                t_hotspot = diags_hotspot_tseqs[i][j]
                score = 0
                for k, char in enumerate(q_hotspot):
                    if type == 'nucleotide':
                        if char == t_hotspot[k]:
                            score += 5
                        else:
                            score -= 4
                    else:
                        score += get_blosum62(char, t_hotspot[k])
                diags_scores[i][j] = score

    top_scores_idx = []
    top_scores = []
    for i, diag in enumerate(diags_scores):
        if diag:
            for j, score in enumerate(diag):
                top_scores_idx.append([i, j])
                top_scores.append(score)

    top_idx = list(np.argsort(top_scores))
    top_idx.reverse()
    sort_top_scores_idx = []
    sort_top_scores = []
    for idx in top_idx:
        if top_scores[idx] > 0:
            sort_top_scores_idx.append(top_scores_idx[idx])
            sort_top_scores.append(top_scores[idx])

    top_N_idx = sort_top_scores_idx[:num_top_diags]

    colors = ['blue', 'purple', 'red', 'peru', 'orange', 'gold', 'yellow', 'lime', 'dodgerblue', 'cyan']
    if plot:
        for i, idx in enumerate(top_N_idx):
            q_start = diags_qhotspots[idx[0]][idx[1]][0]
            t_start = diags_thotspots[idx[0]][idx[1]][0]

            q_end = diags_qhotspots[idx[0]][idx[1]][1]
            t_end = diags_thotspots[idx[0]][idx[1]][1]

            plot[0].plot([t_start, t_end], [q_start, q_end], label=f'HSR{i + 1}', color=colors[i])
            plot[1].plot([t_start, t_end], [q_start, q_end], color=colors[i])

        fig = plot[1].figure
        fig.legend(loc=3, bbox_to_anchor=(0.02, 0.47), fontsize='xx-small', ncol=10)
    # Init One
    init_one_idx = top_N_idx[0]

    chain = [init_one_idx]

    chain_gap = []
    for i, idx in enumerate(top_N_idx[1:]):

        q_start = diags_qhotspots[idx[0]][idx[1]][0]
        t_start = diags_thotspots[idx[0]][idx[1]][0]

        q_end = diags_qhotspots[idx[0]][idx[1]][1]
        t_end = diags_thotspots[idx[0]][idx[1]][1]

        q_chain_start = diags_qhotspots[chain[0][0]][chain[0][1]][0]
        t_chain_start = diags_thotspots[chain[0][0]][chain[0][1]][0]

        q_chain_end = diags_qhotspots[chain[-1][0]][chain[-1][1]][1]
        t_chain_end = diags_thotspots[chain[-1][0]][chain[-1][1]][1]

        if i == 0 and plot:
            plot[2].plot([t_chain_start, t_chain_end], [q_chain_start, q_chain_end], color=colors[i])

        # If end diag run is up and left of the start of the chain
        if q_end <= q_chain_start and t_end <= t_chain_start:
            chain.insert(0, idx)
            x_gap = t_chain_start - t_end
            y_gap = q_chain_start - q_end
            chain_gap.insert(0, max(x_gap, y_gap))
            if plot:
                plot[2].plot([t_start, t_end], [q_start, q_end], color=colors[i + 1])
        # If start diag is down and right of the end of the chain
        elif q_start >= q_chain_end and t_start >= t_chain_end:
            chain.append(idx)
            x_gap = t_start - t_chain_end
            y_gap = q_start - q_chain_end
            chain_gap.append(max(x_gap, y_gap))
            if plot:
                plot[2].plot([t_start, t_end], [q_start, q_end], color=colors[i + 1])
        else:
            for j, i_chain in enumerate(chain[1:]):
                q_left_chain_end = diags_qhotspots[chain[j][0]][chain[j][1]][1]
                t_left_chain_end = diags_thotspots[chain[j][0]][chain[j][1]][1]

                q_right_chain_start = diags_qhotspots[chain[j + 1][0]][chain[j + 1][1]][0]
                t_right_chain_start = diags_thotspots[chain[j + 1][0]][chain[j + 1][1]][0]

                if (q_start >= q_left_chain_end and t_start >= t_left_chain_end) and \
                        (q_end <= q_right_chain_start and t_end <= t_right_chain_start):
                    chain.insert(j, idx)

                    left_x_gap = t_start - t_left_chain_end
                    left_y_gap = q_start - q_left_chain_end

                    right_x_gap = t_right_chain_start - t_end
                    right_y_gap = q_right_chain_start - q_end

                    chain_gap[j - 1] = max(right_x_gap, right_y_gap)
                    chain_gap.insert(j - 1, max(left_x_gap, left_y_gap))
                    if plot:
                        plot[2].plot([t_start, t_end], [q_start, q_end], color=colors[i + 1])

    initn = 0
    for diag, idx in chain:
        initn += diags_scores[diag][idx]

    gaps = len(chain) - 1
    total_gap_pen = gaps * chain_gap_pen

    initn -= total_gap_pen

    return initn


def database_search(query, database_list, type='nucleotide', k_length=6, diag_run_thres=50, diag_allowed_gap=10,
                    diag_gap_pen=-10, num_top_diags=10, chain_gap_pen=40, plot=False):
    handle = Entrez.efetch(db=type, id=query, rettype="gb", retmode="text")
    q_record = SeqIO.read(handle, "genbank")
    handle.close()
    print(f'{q_record.id} \n {q_record.description}')
    query_seq = str(q_record.seq)

    l = float(0.055)
    k = float(0.281)

    initns = []
    Es = []
    # opts = []
    for target in database_list:
        handle = Entrez.efetch(db=type, id=target, rettype="gb", retmode="text")
        record = SeqIO.read(handle, "genbank")
        handle.close()
        print(f'{record.id} \n {record.description}')
        print(f'Align {q_record.id} and {record.id}')
        print(f'Length: {len(q_record)} | Seq: {repr(q_record.seq)}')
        print(f'Length: {len(record)} | Seq: {repr(record.seq)}')
        target_seq = str(record.seq)
        axs = []
        if plot:
            fig = plt.figure()
            gs = gridspec.GridSpec(2, 2)
            gs.update(wspace=0.05, hspace=0.15)  # set the spacing between axes.
            ax1 = plt.subplot(gs[0, 1])  # row 1, span all columns
            ax2 = plt.subplot(gs[0, 0])  # row 0, col 0
            ax3 = plt.subplot(gs[1, 0])
            ax4 = plt.subplot(gs[1, 1])  # row 0, col 1
            ax1.set_title('Sequence Matches', fontsize=6)
            ax2.set_title('Diagonal Runs on Seq Matches', fontsize=6)
            ax3.set_title('Diagonal Runs   ', y=1.0, pad=-10, loc='right', fontsize=6)
            ax4.set_title('Chained Diagonals   ', y=1.0, pad=-10, loc='right', fontsize=6)
            if len(query_seq) > len(target_seq):
                temp = query_seq
                query_seq = target_seq
                target_seq = temp

            mat = np.zeros((len(query_seq), len(target_seq)), dtype=int)
            ax3.imshow(mat, cmap='binary')
            ax4.imshow(mat, cmap='binary')
            for i, el in enumerate(query_seq):
                for j, le in enumerate(target_seq):
                    if el == le:
                        mat[i, j] = 1

            ax1.imshow(mat, cmap='binary')
            ax2.imshow(mat, cmap='binary')
            axs = [ax1, ax3, ax4]

        clock0 = time.time()
        t_initn = FASTA(query_seq, target_seq, type=type, k=k_length, diag_run_thres=diag_run_thres,
                        diag_allowed_gap=diag_allowed_gap, diag_gap_pen=diag_gap_pen, num_top_diags=num_top_diags,
                        chain_gap_pen=chain_gap_pen, plot=axs)

        initns.append(t_initn)
        # opts.append(t_opt)

        m = len(query_seq)
        n = len(target_seq)
        E = 1 - np.exp(-(1 - np.exp(-k * m * n * np.exp(-l * t_initn))) * len(database_list))
        Es.append(E)
        clock1 = time.time()
        print(f' {(clock1 - clock0) * 1000: 2.4f} ms')

        if plot:
            fig.suptitle(f'Initn: {t_initn: 5d}     E-Value: {E: .2E}')
            ax3.set(xlabel=f'{record.id}', ylabel=f'{q_record.id}')

            ax1.set_xticks([])
            ax2.set_xticks([])
            ax2.set_yticks([])
            ax4.set_yticks([])

            ax1.tick_params(labelsize=6)
            ax3.tick_params(labelsize=6)
            ax4.tick_params(labelsize=6)

            if not os.path.exists(f'./{q_record.id}/gap_allow_{diag_allowed_gap}/'):
                os.mkdir(f'./{q_record.id}/gap_allow_{diag_allowed_gap}/')
            plt.savefig(f'./{q_record.id}/gap_allow_{diag_allowed_gap}/target_{record.id}.png')
            if len(database_list) == 1:
                plt.show()
            plt.clf()

    top_initn_idx = list(np.argsort(initns))
    top_initn_idx.reverse()
    top_initns = []
    top_seqs = []
    top_Es = []
    for idx in top_initn_idx:
        top_initns.append(initns[idx])
        top_seqs.append(database_list[idx])
        top_Es.append(Es[idx])

    top_initn = top_initns[0]
    top_seq = top_seqs[0]
    top_E = top_Es[0]

    return top_initn, top_seq, top_E, top_initns, top_seqs, top_Es


if __name__ == '__main__':
    Entrez.email = 'chris.f.angelini@gmail.com'
    '''
    query_acc = "AY707088"
    database_accs = ['NM_001368888.2', 'NM_001368891.2', 'NM_001258465.3', 'NM_000280.5', 'NM_001368887.2',
                     'NM_001258464.2', 'NM_001368889.2', 'NM_001368890.2', 'NM_001127612.3', 'AY707088.1', 'AY047583.1',
                     'BC011953.1', 'M93650.1', 'DQ891436.2', 'DQ894612.2', 'AB528383.1', 'M77844.1', 'AK074881.1',
                     'XM_031015745.1', 'XM_004050882.3','XM_004050881.3', 'XM_004050880.3', 'NM_001368910.2',
                     'NM_001368906.2', 'NM_001368907.2','NM_001368901.2','NM_001368899.2', 'XM_030829473.1',
                     'XM_030829472.1', 'XM_030829471.1', 'XM_030829470.1','XM_030829469.1', 'XM_024255981.1',
                     'XM_024255980.1', 'XM_024255979.1', 'XM_034932347.1', 'XM_024930633.2','XM_003830408.3',
                     'XM_003830407.3', 'XM_003830406.3', 'XM_032166337.1', 'XM_032166336.1', 'XM_032166335.1',
                     'XM_032166334.1','XM_032166333.1', 'XM_025355390.1', 'XM_025355389.1', 'XM_025355388.1',
                     'XM_016920627.2', 'XM_016920625.2','XM_016920626.2', 'XM_016920623.2', 'XM_015434147.1',
                     'XM_015434146.1', 'XM_015434145.1', 'XM_015434144.1','XM_015434143.1', 'XM_012062138.1',
                     'XM_012062137.1', 'XM_012062136.1', 'XM_012062135.1', 'XM_011965593.1','XM_011965592.1',
                     'XM_011965591.1', 'XM_011965590.1', 'XM_011965589.1', 'XM_009186368.3','XM_009186367.3',
                     'XM_009186366.2', 'XM_021926816.2', 'XM_028832522.1', 'XM_028832521.1', 'XM_028832520.1',
                     'XM_015114508.2','XM_015114510.2', 'XM_015114509.2', 'XM_011723992.2', 'XM_011723991.2',
                     'XM_011723990.2', 'XM_024789714.1','XM_011723989.2', 'XM_023230319.1', 'XM_023230318.1',
                     'XM_023230317.1', 'NM_001266257.1','NM_001284987.1','XM_031015752.1', 'XM_019036878.2',
                     'XM_010358333.2', 'XM_010358332.2', 'XM_010358334.2','XM_008003301.2','XM_037997505.1',
                     'XM_008003282.2', 'XM_008003291.2', 'XM_008003272.2', 'XM_030829485.1','XM_030829482.1',
                     'XM_033205274.1']
    type = 'nucleotide'''''

    query_acc = 'AAU12168.1'
    database_accs = ['P26367.2', 'Q1LZF1.1', 'P63016.1', 'P47238.1', 'P55864.1', 'P26630.1', 'O73917.1', 'P47237.1',
                     'G5EDS1.1', 'Q0IH87.2', 'P47239.2', 'P23760.2', 'P24610.2', 'O43316.1', 'P09082.1', 'O88436.1',
                     'P32115.2', 'Q645N4.1', 'P06601.1', 'P23759.4', 'O18381.3', 'Q90268.2', 'O57685.2', 'O57682.2',
                     'Q02962.4', 'P32114.2', 'Q02650.1', 'Q02548.1', 'Q00288.3', 'P09083.2', 'Q9YH95.1', 'P51974.2',
                     'Q06710.2', 'P47240.1', 'Q9PUK5.1', 'A0JMA6.2', 'Q5R9M8.1', 'Q28DP6.2', 'P47236.1', 'Q2VL57.1',
                     'Q2VL59.1', 'Q2VL61.1', 'Q2VL60.1', 'Q2VL58.1', 'Q2VL62.1', 'P55771.3', 'P23757.3', 'Q2L4T2.1',
                     'Q2VL51.1', 'P47242.1', 'Q2VL54.1', 'Q2VL50.1', 'P55166.1', 'Q2VL56.1', 'P09084.4', 'P15863.4',
                     'P23758.3', 'Q9PVX0.1', 'Q9PVY0.1', 'O42358.1', 'O42356.2', 'O42201.2', 'Q06453.2', 'O42567.2',
                     'Q9I9A2.1', 'Q9I9D5.1', 'O35602.2', 'O42357.1', 'Q9JLT7.1', 'O42115.1', 'Q9W2Q1.2', 'Q96IS3.1',
                     'A2T711.1', 'Q96QS3.1', 'A6YP92.1', 'Q9Y2V3.2', 'A6NNA5.1', 'Q8BYH0.2', 'Q7YRX0.1', 'O35085.3',
                     'Q91V10.1', 'Q9IAL2.1', 'O97039.1', 'Q62798.1', 'Q9GMA3.1', 'Q9NZR4.2', 'Q90277.1', 'Q4LAL6.1',
                     'Q94398.3', 'O42250.2', 'Q9H161.2', 'O35137.1', 'Q0P031.1', 'Q26657.2', 'O95076.2', 'O70137.1',
                     'Q1LVQ7.1', 'F1NEA7.2']
    database_accs = ['Q9I9D5.1']
    type = 'protein'

    top_initn, top_seq, E, \
    initn_list, seq_list, E_list = database_search(query_acc, database_accs, type=type,
                                                   k_length=2,
                                                   diag_run_thres=0,
                                                   diag_allowed_gap=30, diag_gap_pen=1,
                                                   num_top_diags=10, chain_gap_pen=11, plot=False)
    print(f'Top Accession: {top_seq}')
    print(f'Top Initn: {top_initn}')
    print(f'Top E-value: {E}')

    print('\nDatabase')
    for i, acc in enumerate(seq_list):
        print(f'Accession: {acc}')
        print(f'\t\tInitn: {initn_list[i]}')
        print(f'\t\tE: {E_list[i]}')
