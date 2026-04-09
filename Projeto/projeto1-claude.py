import numpy as np
import pandas as pd

# ============================================================
# CONSTANTS  [FIX: magic numbers centralised here]
# ============================================================
NETWORK_FACTOR   = 100
COS_PHI          = 0.95
TIME_STEPS       = 200          # AR time-series length
AR_NOISE_STD     = 0.05         # std of complex noise in time series
AR_ANOMALY_FACTOR = 5           # threshold = factor * std(residual)
FROZEN_WINDOW    = 10           # min consecutive flat samples to flag frozen
METER_CORRUPTION = 20           # multiplier for meter-failure simulation
LOAD_SCALE       = 2.0          # abnormal-load scale factor (bus 2)
PF_FACTOR        = 1.5          # power-flow-fault voltage scale
SPIKE_FACTOR     = 3.0          # AR spike magnitude factor
SPIKE_T          = 100          # time step of spike
FROZEN_T         = 100          # time step where freeze begins
HMM_SELF_PROB    = 0.90         # HMM diagonal transition probability
HMM_NOISE_LEVEL  = 0.2          # noise level for noisy classification test
SE_THR_FACTOR    = 30           # SE threshold = factor * median(|r0|)
# [FIX: load-time-step index is explicit, not buried in load_network]
LOAD_TIME_STEP   = 2            # which row of Load(t,Bus) to use as I


# ============================================================
# 0. CARREGAR REDE E CONSTRUIR MATRIZ Y
# ============================================================

def load_network(networkFactor=NETWORK_FACTOR, cosPhi=COS_PHI):
    # [FIX: read Excel once, store all sheets, then extract — avoids reading
    #  the file multiple times]
    xl = pd.ExcelFile('DASG_Prob2_new.xlsx')
    Info      = np.array(xl.parse('Info',        header=None))
    Net_Info  = np.array(xl.parse('Y_Data'))
    Power_raw = np.array(xl.parse('Load(t,Bus)'))

    SlackBus   = int(Info[0, 1])
    Power_Info = np.delete(Power_raw, [0], axis=1)

    P = -Power_Info * np.exp(1j * np.arccos(cosPhi))
    # [FIX: LOAD_TIME_STEP is now a named constant so the choice is explicit]
    I = np.conj(P[LOAD_TIME_STEP, :])

    nBus = int(max(np.max(Net_Info[:, 0]), np.max(Net_Info[:, 1])))
    Y = np.zeros((nBus, nBus), dtype=complex)

    for i in range(Net_Info.shape[0]):
        y_aux = Net_Info[i, 2].replace(",", ".").replace("i", "j")
        y = complex(y_aux) * networkFactor
        a = int(Net_Info[i, 0]) - 1
        b = int(Net_Info[i, 1]) - 1
        Y[a, a] += y
        Y[b, b] += y
        Y[a, b] -= y
        Y[b, a] -= y

    return Y, SlackBus, I


# ============================================================
# HELPER: solve voltages from Y and I (excluding slack)
# ============================================================

def _solve_voltages(Y, SlackBus, I):
    """Return full voltage vector given admittance matrix, slack bus, and
    injected currents (slack bus excluded from the system solve)."""
    nBus = Y.shape[0]
    slack_idx = SlackBus - 1

    Yl = np.delete(np.delete(Y, slack_idx, axis=0), slack_idx, axis=1)

    v = np.zeros(nBus, dtype=complex)
    v[np.arange(nBus) != slack_idx] = 1.0 + np.linalg.solve(Yl, I)
    v[slack_idx] = 1.0
    return v


# ============================================================
# HELPER: build H matrix from admittance matrix
# [FIX: H construction was duplicated in compute_baseline and run_SE;
#  now lives in one place]
# ============================================================

def _build_H(Y):
    """Build the 8×5 measurement Jacobian H for the 5-bus kite grid."""
    y12 = -Y[0, 1]
    y13 = -Y[0, 2]
    y23 = -Y[1, 2]
    y34 = -Y[2, 3]
    y45 = -Y[3, 4]

    Hx = np.zeros((8, 5), dtype=complex)
    # Line current measurements
    Hx[0, 0] =  y12;  Hx[0, 1] = -y12
    Hx[1, 3] = -y45;  Hx[1, 4] =  y45
    # Voltage measurements
    Hx[2, 1] = 1
    Hx[3, 4] = 1
    # Nodal current injections (KCL rows)
    Hx[4, 0] =  y12 + y13;  Hx[4, 1] = -y12;  Hx[4, 2] = -y13
    Hx[5, 0] = -y12;         Hx[5, 1] =  y12 + y23;  Hx[5, 2] = -y23
    Hx[6, 0] = -y13;         Hx[6, 1] = -y23
    Hx[6, 2] =  y13 + y23 + y34;  Hx[6, 3] = -y34
    Hx[7, 2] = -y34;  Hx[7, 3] =  y34 + y45;  Hx[7, 4] = -y45
    return Hx


# ============================================================
# HELPER: weighted-least-squares SE solve
# ============================================================

def _wls_solve(Hx, z):
    """Return estimated state x and residual r = z - H x."""
    HtH = Hx.conj().T @ Hx
    x   = np.linalg.solve(HtH, Hx.conj().T @ z)
    r   = z - Hx @ x
    return x, r


# ============================================================
# 1. PASSO 1 — BASELINE SE
# ============================================================

def compute_baseline(Y, SlackBus, I):
    v0 = _solve_voltages(Y, SlackBus, I)

    y12 = -Y[0, 1]
    y45 = -Y[3, 4]

    z0 = np.zeros(8, dtype=complex)
    z0[0]   = y12 * (v0[0] - v0[1])
    z0[1]   = y45 * (v0[4] - v0[3])
    z0[2]   = v0[1]
    z0[3]   = v0[4]
    z0[4:8] = I

    Hx      = _build_H(Y)
    x0, r0  = _wls_solve(Hx, z0)

    return v0, z0, Hx, x0, r0


# ============================================================
# 2. PASSO 2 — DETEÇÃO DE FALHAS (SE)
# ============================================================

def simulate_line_outage(Y, line):
    """Remove a line from the admittance matrix, correctly updating both
    off-diagonal and diagonal entries.
    [FIX: original only zeroed off-diagonal, leaving diagonal wrong]"""
    a, b = line
    y_line = -Y[a, b]          # recover the admittance of this branch
    Y_fault = Y.copy()
    Y_fault[a, b]  = 0
    Y_fault[b, a]  = 0
    Y_fault[a, a] -= y_line    # remove branch contribution from shunt
    Y_fault[b, b] -= y_line
    return Y_fault


def simulate_meter_failure(z0, index, corruption=METER_CORRUPTION):
    z_fault = z0.copy()
    z_fault[index] *= corruption
    return z_fault


def run_SE(Y, SlackBus, I, z):
    """Run state estimation for an arbitrary measurement vector z.
    [FIX: now uses shared _build_H / _wls_solve helpers instead of
     duplicating the H construction]"""
    Hx     = _build_H(Y)
    x, r   = _wls_solve(Hx, z)
    return x, r


# ============================================================
# 3. PASSO 3 — AUTO-REGRESSÃO (AR)
# ============================================================

def generate_time_series(Y, SlackBus, I, time=TIME_STEPS):
    slack_idx = SlackBus - 1
    Yl = np.delete(np.delete(Y, slack_idx, axis=0), slack_idx, axis=1)
    nFree = Yl.shape[0]

    II    = np.zeros((nFree, time), dtype=complex)
    I123  = np.zeros(time)
    V3    = np.zeros(time)

    II[:, 0] = I
    v = 1.0 + np.linalg.solve(Yl, I)
    I123[0] = np.abs(Y[0, 1] * (v[0] - v[1]))
    V3[0]   = np.abs(v[2])

    # [FIX: noise is now complex so it perturbs both real and imaginary parts
    #  of the current injection symmetrically]
    rng = np.random.default_rng()
    e = (rng.standard_normal(time) + 1j * rng.standard_normal(time)) * AR_NOISE_STD

    for t in range(time - 1):
        II[:, t + 1] = 0.98 * II[:, t] + e[t]
        v = 1.0 + np.linalg.solve(Yl, II[:, t + 1])
        I123[t + 1] = np.abs(Y[0, 1] * (v[0] - v[1]))
        V3[t + 1]   = np.abs(v[2])

    return I123, V3


def fit_AR1(x):
    x_t   = x[1:]
    x_tm1 = x[:-1]
    A     = np.column_stack([x_tm1, np.ones(len(x_tm1))])
    a, b  = np.linalg.lstsq(A, x_t, rcond=None)[0]
    return a, b


def compute_residuals(x, a, b):
    x_hat = a * x[:-1] + b
    r     = x[1:] - x_hat
    return r, x_hat


def detect_anomalies(r, factor=AR_ANOMALY_FACTOR):
    thr       = factor * np.std(r)
    anomalies = np.where(np.abs(r) > thr)[0]
    return anomalies, thr


def detect_frozen_meter(x, window=FROZEN_WINDOW):
    diffs  = np.abs(np.diff(x))
    frozen = np.where(diffs < 1e-4)[0]

    groups, current = [], []
    for idx in frozen:
        if not current or idx == current[-1] + 1:
            current.append(idx)
        else:
            if len(current) >= window:
                groups.append(current)
            current = [idx]
    if len(current) >= window:
        groups.append(current)

    return groups


# ============================================================
# 4. PASSO 4 — SIMULAÇÃO DE FALHAS
# ============================================================

def simulate_abnormal_load(I, bus, scale):
    I_fault      = I.copy()
    I_fault[bus] *= scale
    return I_fault


def simulate_power_flow_fault(Y, SlackBus, I, factor=PF_FACTOR):
    v       = _solve_voltages(Y, SlackBus, I)
    v_fault = v * factor

    y12 = -Y[0, 1]
    y45 = -Y[3, 4]

    z_fault      = np.zeros(8, dtype=complex)
    z_fault[0]   = y12 * (v_fault[0] - v_fault[1])
    z_fault[1]   = y45 * (v_fault[4] - v_fault[3])
    z_fault[2]   = v_fault[1]
    z_fault[3]   = v_fault[4]
    z_fault[4:8] = I
    return z_fault


# ============================================================
# 5. PASSO 5 — RESÍDUOS EM FALHAS & VETORES DE OBSERVAÇÃO
# ============================================================

def simulate_AR_spike(x, t_spike=SPIKE_T, factor=SPIKE_FACTOR):
    x_fault          = x.copy()
    x_fault[t_spike] *= factor
    return x_fault


def simulate_AR_frozen(x, t_start=FROZEN_T):
    x_fault          = x.copy()
    x_fault[t_start:] = x_fault[t_start]
    return x_fault


def residual_features(r):
    """Summarise a residual array as [rms, std, max_abs].
    [FIX: original used only max(|r|), which is fragile to single outliers;
     RMS + std + max gives a richer, more robust feature vector]"""
    abs_r = np.abs(r)
    return np.array([
        np.sqrt(np.mean(abs_r ** 2)),   # RMS
        np.std(abs_r),                  # spread
        np.max(abs_r),                  # peak
    ])


def build_observation_vector(se_residual, ar_I_residual=None, ar_V_residual=None):
    """Build the observation vector fed into the HMM.
    Each residual source contributes 3 features (rms, std, max).
    [FIX: was a single max() per source — now 3 features per source]"""
    parts = [residual_features(se_residual)]
    if ar_I_residual is not None:
        parts.append(residual_features(ar_I_residual))
    if ar_V_residual is not None:
        parts.append(residual_features(ar_V_residual))
    return np.concatenate(parts).real   # observations are real-valued


# ============================================================
# 6. PASSO 6 — HMM PARA CLASSIFICAÇÃO DE FALHAS
#
# Architecture: discrete-time HMM with Gaussian emission.
# Training: one sequence per fault class → estimate mean & covariance
#           of the observation vector for that class.
# Decoding:  Viterbi algorithm over a sequence of observations.
#
# [FIX: original stored one point per state and used nearest-centroid
#  lookup, ignoring the transition matrix A entirely.
#  Now:
#   - hmm_train  stores per-state Gaussian parameters (mean, cov).
#   - hmm_viterbi runs a proper Viterbi decode over a sequence.
#   - hmm_classify_single still works for a single observation
#     (useful for quick checks / noisy tests).]
# ============================================================

def _gaussian_log_likelihood(obs, mean, cov):
    """Log-likelihood of obs under N(mean, cov)."""
    d     = obs - mean
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        return -np.inf
    maha  = d @ np.linalg.solve(cov, d)
    k     = len(obs)
    return -0.5 * (k * np.log(2 * np.pi) + logdet + maha)


def hmm_train(observations_dict, n_fake_samples=50, obs_noise_std=0.05):
    """Estimate HMM parameters from one prototype observation per state.

    Because we have only one real sample per state we synthesise
    n_fake_samples noisy copies to estimate the covariance.

    Returns
    -------
    states : list of str
    A      : (n_states, n_states) transition matrix
    means  : (n_states, d) array of per-state means
    covs   : list of (d, d) covariance matrices
    """
    states   = list(observations_dict.keys())
    n_states = len(states)
    n_obs    = len(next(iter(observations_dict.values())))

    means = np.array([observations_dict[s] for s in states])   # (n_states, d)
    covs  = []

    rng = np.random.default_rng(seed=0)
    for s in states:
        proto  = observations_dict[s]
        samples = proto + rng.standard_normal((n_fake_samples, n_obs)) * obs_noise_std
        covs.append(np.cov(samples.T) + np.eye(n_obs) * 1e-6)   # regularise

    # Transition matrix: high self-loop probability, equal off-diagonal
    off = (1.0 - HMM_SELF_PROB) / max(n_states - 1, 1)
    A   = np.full((n_states, n_states), off)
    np.fill_diagonal(A, HMM_SELF_PROB)

    return states, A, means, covs


def hmm_viterbi(obs_sequence, states, A, means, covs, pi=None):
    """Viterbi decode a sequence of observation vectors.

    Parameters
    ----------
    obs_sequence : (T, d) array — T observations of dimension d
    pi           : (n_states,) initial state distribution (uniform if None)

    Returns
    -------
    best_path  : list of str state labels
    log_probs  : (T, n_states) Viterbi log-probability table
    """
    T        = len(obs_sequence)
    n_states = len(states)
    log_A    = np.log(A + 1e-300)

    if pi is None:
        log_pi = np.full(n_states, -np.log(n_states))
    else:
        log_pi = np.log(pi + 1e-300)

    # Emission log-likelihoods for every (t, state)
    log_B = np.array([
        [_gaussian_log_likelihood(obs_sequence[t], means[s], covs[s])
         for s in range(n_states)]
        for t in range(T)
    ])                                        # (T, n_states)

    # Forward pass
    delta   = np.zeros((T, n_states))
    psi     = np.zeros((T, n_states), dtype=int)
    delta[0] = log_pi + log_B[0]

    for t in range(1, T):
        for s in range(n_states):
            trans = delta[t - 1] + log_A[:, s]
            psi[t, s]   = np.argmax(trans)
            delta[t, s] = trans[psi[t, s]] + log_B[t, s]

    # Back-track
    path = [int(np.argmax(delta[-1]))]
    for t in range(T - 1, 0, -1):
        path.append(psi[t, path[-1]])
    path.reverse()

    return [states[s] for s in path], delta


def hmm_classify_single(obs, states, A, means, covs):
    """Classify a single observation by Gaussian emission likelihood only
    (no sequence context).  Convenient for spot-checks."""
    lls       = np.array([_gaussian_log_likelihood(obs, means[i], covs[i])
                          for i in range(len(states))])
    best      = np.argmax(lls)
    return states[best], lls


# ============================================================
# 7. PLOTTING HELPERS
# [FIX: plotting moved into a dedicated function so __main__ stays clean]
# ============================================================

def plot_results(r0, thr_SE,
                 r_fault, r_fault2,
                 I123, r_I, thr_I,
                 r_I_spike, r_I_frozen,
                 observations,
                 hmm_results):
    import matplotlib.pyplot as plt

    # 1) SE — Baseline
    plt.figure(figsize=(8, 4))
    plt.stem(np.abs(r0))
    plt.axhline(thr_SE, color='red', linestyle='--', label='Threshold')
    plt.title("SE Residuals — Baseline")
    plt.xlabel("Measurement index"); plt.ylabel("|residual|")
    plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

    # 2) SE — Line Outage
    plt.figure(figsize=(8, 4))
    plt.stem(np.abs(r_fault))
    plt.title("SE Residuals — Line Outage")
    plt.xlabel("Measurement index"); plt.ylabel("|residual|")
    plt.grid(True); plt.tight_layout(); plt.show()

    # 3) SE — Meter Failure
    plt.figure(figsize=(8, 4))
    plt.stem(np.abs(r_fault2))
    plt.title("SE Residuals — Meter Failure")
    plt.xlabel("Measurement index"); plt.ylabel("|residual|")
    plt.grid(True); plt.tight_layout(); plt.show()

    # 4) AR time series and residuals
    plt.figure(figsize=(10, 4))
    plt.plot(I123, label="I123 (time series)")
    plt.title("Time Series — I123 (Baseline)")
    plt.xlabel("Time"); plt.ylabel("Current")
    plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(r_I, label="AR(1) Residuals")
    plt.axhline(thr_I,  color='red', linestyle='--', label='AR Threshold')
    plt.axhline(-thr_I, color='red', linestyle='--')
    plt.title("AR(1) Residuals — I123 (Baseline)")
    plt.xlabel("Time"); plt.ylabel("Residual")
    plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(r_I_spike, label="Residuals (Spike)")
    plt.title("AR(1) Residuals — I123 Spike")
    plt.xlabel("Time"); plt.ylabel("Residual")
    plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(r_I_frozen, label="Residuals (Frozen)")
    plt.title("AR(1) Residuals — I123 Frozen")
    plt.xlabel("Time"); plt.ylabel("Residual")
    plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()

    # 5) HMM — Observation Vectors (3 features per source → 9 total)
    states_list = list(observations.keys())
    obs_matrix  = np.array([observations[s] for s in states_list])
    n_feat      = obs_matrix.shape[1]
    labels      = (["SE_rms", "SE_std", "SE_max"] +
                   ["AR_I_rms", "AR_I_std", "AR_I_max"] +
                   ["AR_V_rms", "AR_V_std", "AR_V_max"])[:n_feat]

    x     = np.arange(len(states_list))
    width = 0.8 / n_feat
    plt.figure(figsize=(12, 5))
    for fi in range(n_feat):
        plt.bar(x + fi * width - 0.4, obs_matrix[:, fi], width, label=labels[fi])
    plt.xticks(x, states_list, rotation=20)
    plt.ylabel("Feature value"); plt.title("Observation Vectors per Scenario (HMM)")
    plt.legend(); plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout(); plt.show()

    # 6) HMM classification results
    print("\nHMM — Classification Results")
    for scenario, (state_clean, state_noisy, obs_clean, obs_noisy) in hmm_results.items():
        print(f"\nScenario : {scenario}")
        print(f"  Clean  obs : {obs_clean}")
        print(f"  Classified : {state_clean}")
        print(f"  Noisy  obs : {obs_noisy}")
        print(f"  Classified : {state_noisy}")


# ============================================================
# 8. EXECUTAR PROJECTO COMPLETO
# ============================================================

if __name__ == "__main__":

    # ── 0. Load network ──────────────────────────────────────
    Y, SlackBus, I = load_network()

    # ── 1. Baseline SE ───────────────────────────────────────
    print("\n=== PASSO 1 — BASELINE SE ===")
    v0, z0, Hx, x0, r0 = compute_baseline(Y, SlackBus, I)
    thr_SE = SE_THR_FACTOR * np.median(np.abs(r0))
    print("Threshold SE (heurístico):", thr_SE)
    print("Baseline SE residuals (|r0|):", np.abs(r0))

    # ── 2. SE fault scenarios ────────────────────────────────
    print("\n=== PASSO 2 — LINE OUTAGE 1–2 ===")
    Y_fault    = simulate_line_outage(Y, (0, 1))
    # Re-compute z for the faulted topology so measurements are consistent
    v_fault_lo = _solve_voltages(Y_fault, SlackBus, I)
    y12_f      = -Y_fault[0, 1]          # = 0 after outage
    y45_f      = -Y_fault[3, 4]
    z_fault_lo = np.zeros(8, dtype=complex)
    z_fault_lo[0]   = y12_f * (v_fault_lo[0] - v_fault_lo[1])
    z_fault_lo[1]   = y45_f * (v_fault_lo[4] - v_fault_lo[3])
    z_fault_lo[2]   = v_fault_lo[1]
    z_fault_lo[3]   = v_fault_lo[4]
    z_fault_lo[4:8] = I
    x_fault, r_fault = run_SE(Y_fault, SlackBus, I, z_fault_lo)
    print("Line-outage SE residuals (|r|):", np.abs(r_fault))

    print("\n=== PASSO 2 — METER FAILURE (I12 corrompido) ===")
    z_fault_mf        = simulate_meter_failure(z0, 0, corruption=METER_CORRUPTION)
    x_fault2, r_fault2 = run_SE(Y, SlackBus, I, z_fault_mf)
    print("Meter-failure SE residuals (|r|):", np.abs(r_fault2))

    # ── 3. AR baseline ───────────────────────────────────────
    print("\n=== PASSO 3 — AUTO-REGRESSÃO ===")
    I123, V3 = generate_time_series(Y, SlackBus, I)

    a_I, b_I         = fit_AR1(I123)
    r_I, _           = compute_residuals(I123, a_I, b_I)
    anomalies_I, thr_I = detect_anomalies(r_I)
    frozen_I         = detect_frozen_meter(I123)

    a_V, b_V         = fit_AR1(V3)
    r_V, _           = compute_residuals(V3, a_V, b_V)
    anomalies_V, thr_V = detect_anomalies(r_V)
    frozen_V         = detect_frozen_meter(V3)

    print(f"AR I123 — AR coeff={a_I:.4f}, threshold={thr_I:.6f}, "
          f"anomalies={len(anomalies_I)}, frozen groups={len(frozen_I)}")
    print(f"AR V3   — AR coeff={a_V:.4f}, threshold={thr_V:.6f}, "
          f"anomalies={len(anomalies_V)}, frozen groups={len(frozen_V)}")

    # ── 4. Additional fault scenarios ────────────────────────
    print("\n=== PASSO 4 — ABNORMAL LOAD (bus 2 × 2.0) ===")
    I_abnormal           = simulate_abnormal_load(I, bus=1, scale=LOAD_SCALE)
    v_ab, z_ab, _, x_ab, r_ab = compute_baseline(Y, SlackBus, I_abnormal)
    I123_ab, V3_ab       = generate_time_series(Y, SlackBus, I_abnormal)
    a_I_ab, b_I_ab       = fit_AR1(I123_ab)
    r_I_ab, _            = compute_residuals(I123_ab, a_I_ab, b_I_ab)
    a_V_ab, b_V_ab       = fit_AR1(V3_ab)
    r_V_ab, _            = compute_residuals(V3_ab, a_V_ab, b_V_ab)

    print("\n=== PASSO 4 — POWER FLOW FAULT (factor 1.5) ===")
    z_pf         = simulate_power_flow_fault(Y, SlackBus, I, factor=PF_FACTOR)
    x_pf, r_pf  = run_SE(Y, SlackBus, I, z_pf)

    # ── 5. AR fault scenarios (spike & frozen) ───────────────
    print("\n=== PASSO 5 — AR EM FALHAS (SPIKE & FROZEN) ===")

    I123_spike  = simulate_AR_spike(I123,  t_spike=SPIKE_T, factor=SPIKE_FACTOR)
    r_I_spike, _ = compute_residuals(I123_spike, a_I, b_I)

    V3_spike    = simulate_AR_spike(V3,    t_spike=SPIKE_T, factor=SPIKE_FACTOR)
    r_V_spike, _ = compute_residuals(V3_spike,  a_V, b_V)

    I123_frozen  = simulate_AR_frozen(I123, t_start=FROZEN_T)
    r_I_frozen, _ = compute_residuals(I123_frozen, a_I, b_I)

    V3_frozen    = simulate_AR_frozen(V3,   t_start=FROZEN_T)
    r_V_frozen, _ = compute_residuals(V3_frozen,  a_V, b_V)

    # ── 5b. Build observation vectors ───────────────────────
    observations = {
        "normal"          : build_observation_vector(r0,      r_I,       r_V),
        "line_outage"     : build_observation_vector(r_fault,  r_I,       r_V),
        "meter_failure"   : build_observation_vector(r_fault2, r_I,       r_V),
        "abnormal_load"   : build_observation_vector(r_ab,     r_I_ab,    r_V_ab),
        "power_flow_fault": build_observation_vector(r_pf,     r_I,       r_V),
        "spike"           : build_observation_vector(r0,       r_I_spike, r_V_spike),
        "frozen_meter"    : build_observation_vector(r0,       r_I_frozen,r_V_frozen),
    }

    print("\n=== VETORES DE OBSERVAÇÃO (para HMM) ===")
    for k, v in observations.items():
        print(f"  {k:20s}: {v}")

    # ── 6. HMM training & classification ────────────────────
    print("\n=== PASSO 6 — HMM PARA CLASSIFICAÇÃO DE FALHAS ===")

    states, A, means, covs = hmm_train(observations)

    hmm_results = {}
    rng = np.random.default_rng()

    for scenario, obs in observations.items():
        # Single-observation classification (clean)
        state_clean, _ = hmm_classify_single(obs, states, A, means, covs)

        # Single-observation classification (noisy)
        obs_noisy   = obs + rng.standard_normal(obs.shape) * HMM_NOISE_LEVEL
        state_noisy, _ = hmm_classify_single(obs_noisy, states, A, means, covs)

        hmm_results[scenario] = (state_clean, state_noisy, obs, obs_noisy)

        print(f"\nScenario : {scenario}")
        print(f"  Obs (clean) : {obs}")
        print(f"  Class       : {state_clean}")
        print(f"  Obs (noisy) : {obs_noisy}")
        print(f"  Class       : {state_noisy}")

    # ── Demo of Viterbi over a short sequence ───────────────
    print("\n=== VITERBI DEMO (sequence of 5 observations) ===")
    # Build a short sequence: normal × 3, then spike × 2
    demo_seq = np.array([
        observations["normal"],
        observations["normal"],
        observations["normal"],
        observations["spike"],
        observations["spike"],
    ])
    path, _ = hmm_viterbi(demo_seq, states, A, means, covs)
    print("Input labels : normal, normal, normal, spike, spike")
    print("Viterbi path :", path)

    # ── Uncomment to display plots ───────────────────────────
    plot_results(
         r0, thr_SE,
         r_fault, r_fault2,
         I123, r_I, thr_I,
         r_I_spike, r_I_frozen,
         observations,
         hmm_results)