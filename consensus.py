#!/usr/bin/env python3
"""
DCA + Consensus Sequence Pipeline
==================================
Ports plmDCA_asymmetric (MATLAB) to pure Python using L-BFGS-B,
then computes consensus sequences two ways:
  1. Without DCA  – frequency-based, positions defined by 1STN reference
  2. With DCA     – Potts Hamiltonian greedy energy minimisation, same positions

Reference sequence: the sequence whose header contains '1STN' (first match).
The reference row is tracked explicitly through np.unique so it is never lost.

Usage:
    - Edit USER CONFIG below (FASTAFILE at minimum)
    - Run:  python dca_consensus_pipeline.py
"""

import os
import sys
import time
import numpy as np
from scipy.optimize import minimize
from multiprocessing import Pool


# ============================================================
# USER CONFIG  (edit these)
# ============================================================
FASTAFILE             = "/Users/harry/Desktop/MBL/LA1/Potts_analysis_Sternke_Tripp_Barrick-main/cSN1_mafft_gapStrip.txt"
LAMBDA_J              = 0.01
REWEIGHTING_THRESHOLD = 0.2   # sequences with identity >= 1-0.2 = 80% are clustered
NR_OF_CORES           = 8
OUTPUT_PREFIX         = "dca_output"
REF_ID_SUBSTR         = "1STN"   # substring to match reference sequence header
# ============================================================


# ============================================================
# ALPHABET  (gap=0, A=1 … Y=20, matches MATLAB letter2number)
# ============================================================
AA_ORDER  = "-ACDEFGHIKLMNPQRSTVWY"   # 21 states, index = integer state
AA_TO_IDX = {c: i for i, c in enumerate(AA_ORDER)}

def letter2number(c):
    return AA_TO_IDX.get(c.upper(), 0)   # unknown -> gap (0)


# ============================================================
# FASTA I/O
# ============================================================
def read_fasta(filename):
    """Return list of (header, sequence) tuples."""
    entries, header, seq = [], None, []
    with open(filename) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    entries.append((header, "".join(seq)))
                header, seq = line[1:], []
            else:
                seq.append(line.upper())
    if header is not None:
        entries.append((header, "".join(seq)))
    return entries


def find_ref_index(entries, substr):
    """
    Return the index of the first entry whose header contains *substr*.
    Raises ValueError if not found.
    """
    for i, (hdr, _) in enumerate(entries):
        if substr in hdr:
            return i
    raise ValueError(
        f"Reference sequence '{substr}' not found in alignment headers.\n"
        f"First 5 headers: {[e[0] for e in entries[:5]]}"
    )


# ============================================================
# ALIGNMENT PROCESSING
# ============================================================
def return_alignment(entries, ref_idx):
    """
    Encode alignment as integer matrix Y (B x N).

    Column mask: keep columns that are uppercase and non-'.' in the reference
    sequence — this strips insert columns while keeping ALL 149 reference
    positions (1STN has no gaps, so all 149 are kept).

    Returns
    -------
    N        : number of alignment columns kept
    B        : number of sequences
    q        : max state value (= alphabet size - 1, typically 20)
    Y        : (B, N) int32 array
    """
    ref_seq = entries[ref_idx][1]
    keep_cols = [j for j, c in enumerate(ref_seq) if c != "." and c.isupper()]
    N = len(keep_cols)
    B = len(entries)

    Y = np.zeros((B, N), dtype=np.int32)
    for i, (_, seq) in enumerate(entries):
        for ci, j in enumerate(keep_cols):
            if j < len(seq):
                Y[i, ci] = letter2number(seq[j])

    q = int(Y.max())
    return N, B, q, Y


def deduplicate(Y, ref_row):
    """
    Remove duplicate rows from Y with np.unique, but track where ref_row ends up.

    Returns
    -------
    Y_unique   : deduplicated matrix
    ref_idx_new: row index of the reference sequence in Y_unique
    """
    Y_unique, orig_idx = np.unique(Y, axis=0, return_index=True)

    # Find the unique row that matches the original reference row
    ref_vec   = Y[ref_row]
    matches   = np.where((Y_unique == ref_vec).all(axis=1))[0]
    if len(matches) == 0:
        raise RuntimeError("Reference sequence was lost during deduplication — this should not happen.")
    ref_idx_new = int(matches[0])

    return Y_unique, ref_idx_new, orig_idx


# ============================================================
# SEQUENCE REWEIGHTING  (mirrors calc_inverse_weights.c)
# ============================================================
def calc_inverse_weights(Y, threshold):
    """
    weight_i = 1 / #{j : identity(i,j) >= 1 - threshold}
    Vectorised in chunks to avoid OOM on large alignments.
    """
    B, N = Y.shape
    print("  Computing pairwise sequence identities...")
    m = np.ones(B, dtype=np.float64)
    chunk = 500
    for i in range(0, B, chunk):
        end  = min(i + chunk, B)
        frac = (Y[i:end, None, :] == Y[None, :, :]).mean(axis=2)   # (chunk, B)
        m[i:end] += (frac >= (1.0 - threshold)).sum(axis=1) - 1     # subtract self
    return 1.0 / m


# ============================================================
# PSEUDOLIKELIHOOD OBJECTIVE  (mirrors g_rC.c, L2 regularisation)
# ============================================================
def g_r_objective(wr, Y, weights, N, q, lambda_h, lambda_J, r):
    """
    Regularised pseudolikelihood value + gradient for node r (0-based).

    States 0..q  (nq = q+1 total states; 0 = gap).
    Parameter vector wr has length  nq + nq^2*(N-1).
    """
    nq  = int(q) + 1
    B   = Y.shape[0]
    r   = int(r)

    h_r = wr[:nq]                           # (nq,)
    J_r = wr[nq:].reshape(nq, nq, N - 1)   # (nq, nq, N-1)

    # --- log-potentials  (B x nq) ---
    logPot     = np.tile(h_r, (B, 1))
    other_cols = [n for n in range(N) if n != r]
    for ci, n in enumerate(other_cols):
        y2 = Y[:, n].astype(int)
        logPot += J_r[:, y2, ci].T          # broadcast over sequences

    # --- stable log-sum-exp ---
    lp_max  = logPot.max(axis=1, keepdims=True)
    exp_lp  = np.exp(logPot - lp_max)
    Z       = exp_lp.sum(axis=1)
    log_Z   = np.log(Z) + lp_max[:, 0]

    y1 = Y[:, r].astype(int)
    fval = float(
        -np.dot(weights, logPot[np.arange(B), y1]) + np.dot(weights, log_Z)
    )

    # --- node beliefs ---
    nodeBel = exp_lp / Z[:, None]           # (B, nq)

    # --- gradient h ---
    grad_h = np.zeros(nq)
    np.add.at(grad_h, y1, -weights)
    grad_h += (weights[:, None] * nodeBel).sum(axis=0)

    # --- gradient J ---
    grad_J = np.zeros((nq, nq, N - 1))
    for ci, n in enumerate(other_cols):
        y2 = Y[:, n].astype(int)
        np.add.at(grad_J[:, :, ci], (y1, y2), -weights)
        for s in range(nq):
            np.add.at(grad_J[s, :, ci], y2, weights * nodeBel[:, s])

    # --- L2 regularisation ---
    fval   += lambda_h * np.dot(h_r, h_r)
    grad_h += 2.0 * lambda_h * h_r
    fval   += lambda_J * np.sum(J_r ** 2)
    grad_J += 2.0 * lambda_J * J_r

    return fval, np.concatenate([grad_h, grad_J.ravel()])


def min_g_r(args):
    """Minimise g_r for one node; suitable as a multiprocessing worker."""
    Y, weights, N, q, lh, lJ, r, _ = args
    nq  = int(q) + 1
    wr0 = np.zeros(nq + nq ** 2 * (int(N) - 1))

    print(f"  Minimising node r={r+1}/{N}")
    res = minimize(
        lambda wr: g_r_objective(wr, Y, weights, int(N), int(q), lh, lJ, r),
        wr0,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 500, "ftol": 1e-9, "gtol": 1e-6, "disp": False},
    )
    return r, res.x


# ============================================================
# GAUGE TRANSFORMS
# ============================================================
def to_ising_gauge(J_raw):
    """Zero-sum (Ising) gauge for one q×q coupling matrix."""
    mu   = J_raw.mean()
    mu_r = J_raw.mean(axis=1, keepdims=True)
    mu_c = J_raw.mean(axis=0, keepdims=True)
    return J_raw - mu_r - mu_c + mu


def h_zero_sum_gauge(h, J_full):
    """
    Shift h into zero-sum gauge (Ekeberg 2014, Eq. 24).
    h     : (N, nq)
    J_full: (N, N, nq, nq)
    """
    N, nq = h.shape
    h_hat = h - h.mean(axis=1, keepdims=True)
    for i in range(N):
        for j in range(N):
            if i != j:
                Jij       = J_full[i, j]            # (nq, nq)
                full_mean = Jij.mean()
                row_means = Jij.mean(axis=1)         # (nq,) mean over column states
                h_hat[i] += row_means - full_mean
    return h_hat


# ============================================================
# APC-CORRECTED FROBENIUS NORMS
# ============================================================
def compute_di_scores(J_gauge, N):
    """
    J_gauge: (n_pairs, nq, nq)  gauge-transformed couplings.
    Returns CORRNORMS (N x N) APC-corrected DI score matrix.
    """
    NORMS = np.zeros((N, N))
    l = 0
    for i in range(N - 1):
        for j in range(i + 1, N):
            fn          = np.linalg.norm(J_gauge[l, 1:, 1:], "fro")   # skip gap
            NORMS[i, j] = fn
            NORMS[j, i] = fn
            l += 1
    row_means   = NORMS.mean(axis=1) * N / (N - 1)
    global_mean = NORMS.mean()        * N / (N - 1)
    CORRNORMS   = NORMS - np.outer(row_means, row_means) / global_mean
    return CORRNORMS


# ============================================================
# CONSENSUS WITHOUT DCA  (frequency-based, reference-guided)
# ============================================================
def consensus_no_dca(Y, ref_row):
    """
    For every alignment column that is *occupied* (non-gap) in the reference
    sequence, take the most frequent non-gap residue across the full MSA.

    Parameters
    ----------
    Y       : (B, N) integer alignment matrix
    ref_row : row index of the reference (1STN) sequence in Y

    Returns
    -------
    consensus_str  : string over AA_ORDER alphabet
    ref_positions  : list of column indices (0-based) used
    """
    ref_seq        = Y[ref_row]
    ref_positions  = [col for col in range(Y.shape[1]) if ref_seq[col] != 0]

    consensus_chars = []
    for col in ref_positions:
        counts       = np.bincount(Y[:, col], minlength=21).astype(float)
        counts[0]    = 0.0                 # zero out gap votes
        best_state   = int(counts.argmax())
        consensus_chars.append(AA_ORDER[best_state] if counts[best_state] > 0 else "-")

    return "".join(consensus_chars), ref_positions


# ============================================================
# CONSENSUS WITH DCA  (greedy Potts energy, reference-guided)
# ============================================================
def consensus_with_dca(h_gauge, J_full, Y, ref_row):
    """
    Greedy Potts Hamiltonian consensus.

    At each reference-occupied position (in left-to-right order), choose the
    non-gap amino-acid state that minimises the local energy:
        E(a) = -h[col, a]  -  sum_{j already chosen} J[col, j, a, chosen[j]]

    Parameters
    ----------
    h_gauge : (N, nq)       zero-sum fields
    J_full  : (N, N, nq, nq) zero-sum couplings (full symmetric tensor)
    Y       : (B, N)        integer alignment matrix
    ref_row : row index of the reference (1STN) sequence in Y

    Returns
    -------
    consensus_str : string over AA_ORDER alphabet
    ref_positions : list of column indices used (same as no-DCA)
    """
    N, nq     = h_gauge.shape
    ref_seq   = Y[ref_row]
    ref_positions = [col for col in range(N) if ref_seq[col] != 0]

    chosen = {}   # col -> chosen integer state

    for col in ref_positions:
        energies = -h_gauge[col].copy()            # (nq,) — contribution from fields
        for prev_col, prev_state in chosen.items():
            energies -= J_full[col, prev_col, :, prev_state]

        # Force non-gap: consider states 1..nq-1 only
        best_state     = int(energies[1:].argmin()) + 1
        chosen[col]    = best_state

    consensus_chars = [AA_ORDER[chosen[col]] for col in ref_positions]
    return "".join(consensus_chars), ref_positions


# ============================================================
# UTILITIES
# ============================================================
def marginal_frequencies(Y, positions, nstates=21):
    B      = Y.shape[0]
    freqs  = np.zeros((len(positions), nstates))
    for pi, col in enumerate(positions):
        counts      = np.bincount(Y[:, col], minlength=nstates).astype(float)
        freqs[pi]   = counts / B
    return freqs


def sequence_entropy(freqs):
    """Shannon entropy in bits."""
    eps = 1e-12
    return -(freqs * np.log2(freqs + eps)).sum(axis=1)


# ============================================================
# MAIN PIPELINE
# ============================================================
def run_pipeline(fastafile, lambda_J=0.01, reweighting_threshold=0.2,
                 nr_of_cores=1, output_prefix="dca_output",
                 ref_id_substr="1STN"):

    os.makedirs(output_prefix, exist_ok=True)
    sep = "=" * 60

    print(f"\n{sep}")
    print("DCA + Consensus Pipeline")
    print(f"Input    : {fastafile}")
    print(f"Output   : {output_prefix}/")
    print(f"Reference: sequences containing '{ref_id_substr}'")
    print(f"lambda_J={lambda_J}  reweight_threshold={reweighting_threshold}  cores={nr_of_cores}")
    print(f"{sep}\n")

    # ------------------------------------------------------------------
    # 1. Read alignment  +  locate reference sequence
    # ------------------------------------------------------------------
    print("[1/6] Reading alignment...")
    entries = read_fasta(fastafile)

    # Find the reference row BEFORE any deduplication
    ref_row_raw = find_ref_index(entries, ref_id_substr)
    print(f"  Found reference '{entries[ref_row_raw][0]}' at row {ref_row_raw}")

    N, B_raw, q, Y = return_alignment(entries, ref_row_raw)
    print(f"  Sequences: {B_raw}  Alignment length: {N}  q (max state): {q}")

    # Verify reference has the expected number of non-gap positions
    ref_positions_raw = [col for col in range(N) if Y[ref_row_raw, col] != 0]
    print(f"  Reference non-gap positions: {len(ref_positions_raw)} / {N}")

    # ------------------------------------------------------------------
    # 2. Deduplicate — tracking the reference row
    # ------------------------------------------------------------------
    print("\n[2/6] Deduplicating sequences...")
    Y_u, ref_row, _ = deduplicate(Y, ref_row_raw)
    B = Y_u.shape[0]
    print(f"  Unique sequences: {B}  (removed {B_raw - B} duplicates)")
    print(f"  Reference row in deduplicated matrix: {ref_row}")

    # Double-check: reference positions must still be 149 non-gap cols
    ref_positions_check = [col for col in range(N) if Y_u[ref_row, col] != 0]
    print(f"  Reference non-gap positions after dedup: {len(ref_positions_check)} / {N}")
    assert len(ref_positions_check) == len(ref_positions_raw), \
        "Reference sequence changed after deduplication!"

    np.save(os.path.join(output_prefix, "msa_seq_vectors.npy"), Y_u)

    # ------------------------------------------------------------------
    # 3. Sequence reweighting
    # ------------------------------------------------------------------
    print(f"\n[3/6] Sequence reweighting (threshold={reweighting_threshold})...")
    t0 = time.time()
    if reweighting_threshold > 0.0:
        weights = calc_inverse_weights(Y_u, reweighting_threshold)
    else:
        weights = np.ones(B, dtype=np.float64)
    B_eff = weights.sum()
    print(f"  B_eff = {B_eff:.2f}  ({time.time()-t0:.1f}s)")
    np.save(os.path.join(output_prefix, "seq_weights.npy"), weights)

    # ------------------------------------------------------------------
    # 4. plmDCA optimisation
    # ------------------------------------------------------------------
    nq          = q + 1           # 0=gap, 1..q=amino acids
    scaled_lh   = lambda_J  * B_eff
    scaled_lJ   = lambda_J  * B_eff / 2.0   # /2 to match symmetric variant

    print(f"\n[4/6] plmDCA optimisation ({N} nodes, {nr_of_cores} core(s))...")
    print(f"  scaled_lambda_h={scaled_lh:.3f}  scaled_lambda_J={scaled_lJ:.3f}")
    t0 = time.time()

    tasks = [
        (Y_u, weights, N, q, scaled_lh, scaled_lJ, r, {})
        for r in range(N)
    ]

    w = np.zeros((nq + nq**2 * (N - 1), N))   # parameter matrix (columns = nodes)
    if nr_of_cores > 1:
        with Pool(nr_of_cores) as pool:
            for r_idx, wr in pool.imap_unordered(min_g_r, tasks):
                w[:, r_idx] = wr
    else:
        for task in tasks:
            r_idx, wr = min_g_r(task)
            w[:, r_idx] = wr

    print(f"  Done ({time.time()-t0:.1f}s)")

    h_raw = w[:nq, :].T   # (N, nq)  — raw fields, not yet gauge-transformed
    np.save(os.path.join(output_prefix, "h_raw.npy"), h_raw)

    # ------------------------------------------------------------------
    # 5. Extract J, apply Ising gauge, symmetrise
    # ------------------------------------------------------------------
    print("\n[5/6] Extracting couplings + Ising gauge...")
    JJ       = w[nq:, :].reshape(nq, nq, N - 1, N)
    n_pairs  = N * (N - 1) // 2
    pair_idx = [(i, j) for i in range(N - 1) for j in range(i + 1, N)]

    Jtemp1 = np.zeros((n_pairs, nq, nq))
    Jtemp2 = np.zeros((n_pairs, nq, nq))
    for l, (i, j) in enumerate(pair_idx):
        Jtemp1[l] = JJ[:, :, j - 1, i]      # J_ij from g_i
        Jtemp2[l] = JJ[:, :, i, j].T         # J_ij from g_j

    J_gauge = np.zeros_like(Jtemp1)
    for l in range(n_pairs):
        J_gauge[l] = 0.5 * (to_ising_gauge(Jtemp1[l]) + to_ising_gauge(Jtemp2[l]))

    # Build full symmetric (N, N, nq, nq) tensor
    J_full = np.zeros((N, N, nq, nq))
    for l, (i, j) in enumerate(pair_idx):
        J_full[i, j] = J_gauge[l]
        J_full[j, i] = J_gauge[l].T

    np.save(os.path.join(output_prefix, "J_gauge.npy"), J_gauge)
    np.save(os.path.join(output_prefix, "Jtemp1.npy"),  Jtemp1)
    np.save(os.path.join(output_prefix, "Jtemp2.npy"),  Jtemp2)

    # h zero-sum gauge (average over the two asymmetric estimates)
    J_full1 = np.zeros((N, N, nq, nq))
    J_full2 = np.zeros((N, N, nq, nq))
    for l, (i, j) in enumerate(pair_idx):
        J_full1[i, j] = to_ising_gauge(Jtemp1[l])
        J_full1[j, i] = to_ising_gauge(Jtemp1[l]).T
        J_full2[i, j] = to_ising_gauge(Jtemp2[l])
        J_full2[j, i] = to_ising_gauge(Jtemp2[l]).T

    h_gauge = 0.5 * (
        h_zero_sum_gauge(h_raw.copy(), J_full1) +
        h_zero_sum_gauge(h_raw.copy(), J_full2)
    )
    np.save(os.path.join(output_prefix, "h_gauge.npy"), h_gauge)

    # APC-corrected DI scores
    CORRNORMS = compute_di_scores(J_gauge, N)
    top_pairs = sorted(
        [(i+1, j+1, CORRNORMS[i, j]) for i, j in pair_idx],
        key=lambda x: -x[2]
    )
    with open(os.path.join(output_prefix, "dca_pair_scores.txt"), "w") as fh:
        for i, j, sc in top_pairs:
            fh.write(f"{i} {j} {sc:.6f}\n")
    print(f"  Top DI pair: {top_pairs[0][0]}-{top_pairs[0][1]}  score={top_pairs[0][2]:.4f}")

    # ------------------------------------------------------------------
    # 6. Consensus sequences  (both use 1STN-guided positions)
    # ------------------------------------------------------------------
    print("\n[6/6] Computing consensus sequences...")

    # --- frequency-based (no DCA) ---
    cons_nodca, ref_pos = consensus_no_dca(Y_u, ref_row)
    print(f"\n  Consensus WITHOUT DCA  ({len(cons_nodca)} residues):")
    print(f"  {cons_nodca}")

    # --- Potts energy-guided (DCA) ---
    cons_dca, _ = consensus_with_dca(h_gauge, J_full, Y_u, ref_row)
    print(f"\n  Consensus WITH DCA     ({len(cons_dca)} residues):")
    print(f"  {cons_dca}")

    # reference sequence itself (for three-way comparison)
    ref_decoded = "".join(AA_ORDER[Y_u[ref_row, col]] for col in ref_pos)
    print(f"\n  Reference (1STN)       ({len(ref_decoded)} residues):")
    print(f"  {ref_decoded}")

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------
    assert len(cons_nodca) == len(cons_dca) == len(ref_decoded), \
        "Consensus lengths disagree — bug!"

    n_pos   = len(cons_nodca)
    n_agree = sum(a == b for a, b in zip(cons_nodca, cons_dca))

    # Per-position identity with 1STN
    id_nodca = sum(a == r for a, r in zip(cons_nodca, ref_decoded))
    id_dca   = sum(a == r for a, r in zip(cons_dca,   ref_decoded))

    print(f"\n{sep}")
    print("COMPARISON")
    print(f"{sep}")
    print(f"  Positions:                          {n_pos}")
    print(f"  No-DCA vs DCA in agreement:         {n_agree} ({100*n_agree/n_pos:.1f}%)")
    print(f"  No-DCA identity to 1STN:            {id_nodca} ({100*id_nodca/n_pos:.1f}%)")
    print(f"  DCA    identity to 1STN:            {id_dca}  ({100*id_dca/n_pos:.1f}%)")

    diffs = [
        (i + 1, ref_decoded[i], cons_nodca[i], cons_dca[i])
        for i in range(n_pos)
        if cons_nodca[i] != cons_dca[i]
    ]
    if diffs:
        print(f"\n  Pos | 1STN | No-DCA | DCA")
        print(f"  ----|------|--------|----")
        for pos, ref_aa, aa_nd, aa_d in diffs[:40]:
            print(f"  {pos:4d} | {ref_aa:4s} | {aa_nd:6s} | {aa_d}")
        if len(diffs) > 40:
            print(f"  ... and {len(diffs)-40} more")

    freqs = marginal_frequencies(Y_u, ref_pos)
    H     = sequence_entropy(freqs)
    print(f"\n  Mean positional entropy: {H.mean():.3f} bits")

    # ------------------------------------------------------------------
    # Save text report
    # ------------------------------------------------------------------
    out_txt = os.path.join(output_prefix, "consensus_comparison.txt")
    with open(out_txt, "w") as fh:
        fh.write("DCA + Consensus Pipeline Results\n")
        fh.write(f"Input: {fastafile}\n")
        fh.write(f"Reference: {entries[ref_row_raw][0]}\n")
        fh.write(f"Alignment: {B_raw} seqs (unique: {B}), length {N}, q={q}\n")
        fh.write(f"B_eff: {B_eff:.2f}\n")
        fh.write(f"lambda_J: {lambda_J}  reweight_threshold: {reweighting_threshold}\n\n")

        fh.write(f"{sep}\nREFERENCE (1STN, {len(ref_decoded)} residues)\n{sep}\n")
        fh.write(f"{ref_decoded}\n\n")

        fh.write(f"{sep}\nCONSENSUS WITHOUT DCA (frequency, reference-guided, {len(cons_nodca)} residues)\n{sep}\n")
        fh.write(f"{cons_nodca}\n\n")

        fh.write(f"{sep}\nCONSENSUS WITH DCA (Potts energy, reference-guided, {len(cons_dca)} residues)\n{sep}\n")
        fh.write(f"{cons_dca}\n\n")

        fh.write(f"{sep}\nCOMPARISON\n{sep}\n")
        fh.write(f"Positions:                 {n_pos}\n")
        fh.write(f"No-DCA vs DCA agreement:   {n_agree} ({100*n_agree/n_pos:.1f}%)\n")
        fh.write(f"No-DCA identity to 1STN:   {id_nodca} ({100*id_nodca/n_pos:.1f}%)\n")
        fh.write(f"DCA identity to 1STN:      {id_dca}  ({100*id_dca/n_pos:.1f}%)\n\n")
        fh.write(f"Mean positional entropy: {H.mean():.3f} bits\n\n")
        fh.write("Pos | 1STN | No-DCA | DCA\n")
        fh.write("----|------|--------|----\n")
        for pos, ref_aa, aa_nd, aa_d in diffs:
            fh.write(f"{pos:4d} | {ref_aa:4s} | {aa_nd:6s} | {aa_d}\n")

    print(f"\n  Saved: {out_txt}")
    print(f"  Arrays in: {output_prefix}/")
    print(f"\n{sep}")
    print("Pipeline complete.")
    print(f"{sep}\n")

    return dict(
        consensus_no_dca=cons_nodca,
        consensus_dca=cons_dca,
        ref_sequence=ref_decoded,
        positions=ref_pos,
        h_gauge=h_gauge,
        J_gauge=J_gauge,
        J_full=J_full,
        CORRNORMS=CORRNORMS,
        weights=weights,
        Y=Y_u,
        ref_row=ref_row,
    )


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    if FASTAFILE.startswith("/Users/harry"):
        # Running on Harry's local machine — use the config above as-is
        pass
    run_pipeline(
        fastafile             = FASTAFILE,
        lambda_J              = LAMBDA_J,
        reweighting_threshold = REWEIGHTING_THRESHOLD,
        nr_of_cores           = NR_OF_CORES,
        output_prefix         = OUTPUT_PREFIX,
        ref_id_substr         = REF_ID_SUBSTR,
    )