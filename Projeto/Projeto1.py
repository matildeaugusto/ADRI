import numpy as np
import pandas as pd

# ============================================================
# 0. CARREGAR REDE E CONSTRUIR MATRIZ Y
# ============================================================

def load_network(networkFactor=100, cosPhi=0.95):
    Info = np.array(pd.read_excel('DASG_Prob2_new.xlsx', sheet_name='Info', header=None))
    SlackBus = Info[0,1]

    Net_Info = np.array(pd.read_excel('DASG_Prob2_new.xlsx', sheet_name='Y_Data'))
    Power_Info = np.array(pd.read_excel('DASG_Prob2_new.xlsx', sheet_name='Load(t,Bus)'))
    Power_Info = np.delete(Power_Info, [0], axis=1)

    P = -Power_Info * np.exp(1j * np.arccos(cosPhi))
    I = np.conj(P[2,:])

    nBus = int(max(np.max(Net_Info[:,0]), np.max(Net_Info[:,1])))
    Y = np.zeros((nBus, nBus), dtype=complex)

    for i in range(Net_Info.shape[0]):
        y_aux = Net_Info[i,2].replace(",",".").replace("i","j")
        y = complex(y_aux) * networkFactor
        a = Net_Info[i,0] - 1
        b = Net_Info[i,1] - 1
        Y[a,a] += y
        Y[b,b] += y
        Y[a,b] -= y
        Y[b,a] -= y

    return Y, SlackBus, I


# ============================================================
# 1. PASSO 1 — BASELINE SE
# ============================================================

def compute_baseline(Y, SlackBus, I):
    nBus = Y.shape[0]

    Yl = np.delete(Y, SlackBus-1, axis=0)
    Yl = np.delete(Yl, SlackBus-1, axis=1)

    v0 = np.zeros(nBus, dtype=complex)
    v0[0:nBus-1] = 1 + np.linalg.inv(Yl) @ I
    v0[SlackBus-1] = 1

    y12 = -Y[0,1]
    y45 = -Y[3,4]

    z0 = np.zeros(8, dtype=complex)
    z0[0] = y12 * (v0[0] - v0[1])
    z0[1] = y45 * (v0[4] - v0[3])
    z0[2] = v0[1]
    z0[3] = v0[4]
    z0[4:8] = I

    Hx = np.zeros((8,5), dtype=complex)

    y13 = -Y[0,2]
    y23 = -Y[1,2]
    y34 = -Y[2,3]

    Hx[0,0] = y12
    Hx[0,1] = -y12
    Hx[1,3] = -y45
    Hx[1,4] = y45
    Hx[2,1] = 1
    Hx[3,4] = 1
    Hx[4,0] = y12 + y13
    Hx[4,1] = -y12
    Hx[4,2] = -y13
    Hx[5,0] = -y12
    Hx[5,1] = y12 + y23
    Hx[5,2] = -y23
    Hx[6,0] = -y13
    Hx[6,1] = -y23
    Hx[6,2] = y13 + y23 + y34
    Hx[6,3] = -y34
    Hx[7,2] = -y34
    Hx[7,3] = y34 + y45
    Hx[7,4] = -y45

    x0 = np.linalg.inv(Hx.conj().T @ Hx) @ (Hx.conj().T @ z0)
    r0 = z0 - Hx @ x0

    return v0, z0, Hx, x0, r0


# ============================================================
# 2. PASSO 2 — DETEÇÃO DE FALHAS (SE)
# ============================================================

def simulate_line_outage(Y, line):
    a, b = line
    Y_fault = Y.copy()
    Y_fault[a,b] = 0
    Y_fault[b,a] = 0
    return Y_fault

def simulate_meter_failure(z0, index, corruption=20):
    z_fault = z0.copy()
    z_fault[index] *= corruption
    return z_fault

def run_SE(Y, SlackBus, I, z):
    nBus = Y.shape[0]

    Yl = np.delete(Y, SlackBus-1, axis=0)
    Yl = np.delete(Yl, SlackBus-1, axis=1)

    v = np.zeros(nBus, dtype=complex)
    v[0:nBus-1] = 1 + np.linalg.inv(Yl) @ I
    v[SlackBus-1] = 1

    y12 = -Y[0,1]
    y45 = -Y[3,4]
    y13 = -Y[0,2]
    y23 = -Y[1,2]
    y34 = -Y[2,3]

    Hx = np.zeros((8,5), dtype=complex)

    Hx[0,0] = y12
    Hx[0,1] = -y12
    Hx[1,3] = -y45
    Hx[1,4] = y45
    Hx[2,1] = 1
    Hx[3,4] = 1
    Hx[4,0] = y12 + y13
    Hx[4,1] = -y12
    Hx[4,2] = -y13
    Hx[5,0] = -y12
    Hx[5,1] = y12 + y23
    Hx[5,2] = -y23
    Hx[6,0] = -y13
    Hx[6,1] = -y23
    Hx[6,2] = y13 + y23 + y34
    Hx[6,3] = -y34
    Hx[7,2] = -y34
    Hx[7,3] = y34 + y45
    Hx[7,4] = -y45

    x = np.linalg.inv(Hx.conj().T @ Hx) @ (Hx.conj().T @ z)
    r = z - Hx @ x

    return x, r


# ============================================================
# 3. PASSO 3 — AUTO‑REGRESSÃO (AR)
# ============================================================

def generate_time_series(Y, SlackBus, I, time=200):
    Yl = np.delete(Y, SlackBus-1, axis=0)
    Yl = np.delete(Yl, SlackBus-1, axis=1)

    II = np.zeros((Yl.shape[0], time), dtype=complex)
    I123 = np.zeros(time)
    V3 = np.zeros(time)

    II[:,0] = I
    v = 1 + np.linalg.inv(Yl) @ I
    I123[0] = np.abs(Y[0,1] * (v[0] - v[1]))
    V3[0] = np.abs(v[2])

    e = np.random.randn(time) * 0.05

    for t in range(time-1):
        II[:,t+1] = 0.98 * II[:,t] + e[t]
        v = 1 + np.linalg.inv(Yl) @ II[:,t+1]

        I123[t+1] = np.abs(Y[0,1] * (v[0] - v[1]))
        V3[t+1] = np.abs(v[2])

    return I123, V3


def fit_AR1(x):
    x_t = x[1:]
    x_tm1 = x[:-1]
    A = np.vstack([x_tm1, np.ones(len(x_tm1))]).T
    a, b = np.linalg.lstsq(A, x_t, rcond=None)[0]
    return a, b


def compute_residuals(x, a, b):
    x_hat = a * x[:-1] + b
    r = x[1:] - x_hat
    return r, x_hat


def detect_anomalies(r, factor=5):
    thr = factor * np.std(r)
    anomalies = np.where(np.abs(r) > thr)[0]
    return anomalies, thr


def detect_frozen_meter(x, window=10):
    diffs = np.abs(np.diff(x))
    frozen = np.where(diffs < 1e-4)[0]

    groups = []
    current = []

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
    I_fault = I.copy()
    I_fault[bus] *= scale
    return I_fault

def simulate_power_flow_fault(Y, SlackBus, I, factor=1.5):
    nBus = Y.shape[0]

    Yl = np.delete(Y, SlackBus-1, axis=0)
    Yl = np.delete(Yl, SlackBus-1, axis=1)

    v = np.zeros(nBus, dtype=complex)
    v[0:nBus-1] = 1 + np.linalg.inv(Yl) @ I
    v[SlackBus-1] = 1

    v_fault = v * factor

    y12 = -Y[0,1]
    y45 = -Y[3,4]

    z_fault = np.zeros(8, dtype=complex)
    z_fault[0] = y12 * (v_fault[0] - v_fault[1])
    z_fault[1] = y45 * (v_fault[4] - v_fault[3])
    z_fault[2] = v_fault[1]
    z_fault[3] = v_fault[4]
    z_fault[4:8] = I

    return z_fault


# ============================================================
# 5. PASSO 5 — RESÍDUOS EM FALHAS & VETORES HMM
# ============================================================

def simulate_AR_spike(x, t_spike=100, factor=3.0):
    x_fault = x.copy()
    x_fault[t_spike] *= factor
    return x_fault

def simulate_AR_frozen(x, t_start=100):
    x_fault = x.copy()
    x_fault[t_start:] = x_fault[t_start]
    return x_fault

def max_abs_complex(v):
    return np.max(np.abs(v))

def build_observation_vector(se_residual, ar_I_residual=None, ar_V_residual=None):
    obs = [max_abs_complex(se_residual)]
    if ar_I_residual is not None:
        obs.append(np.max(np.abs(ar_I_residual)))
    if ar_V_residual is not None:
        obs.append(np.max(np.abs(ar_V_residual)))
    return np.array(obs)


# ============================================================
# 6. PASSO 6 — HMM PARA CLASSIFICAÇÃO DE FALHAS
# ============================================================

def hmm_train(observations_dict):
    states = list(observations_dict.keys())
    n_states = len(states)

    A = np.eye(n_states) * 0.90 + (np.ones((n_states, n_states)) - np.eye(n_states)) * 0.025
    means = np.array([observations_dict[s] for s in states])

    return states, A, means

def hmm_emission_likelihood(obs, mean):
    dist = np.linalg.norm(obs - mean)
    return np.exp(-dist)

def hmm_classify(obs, states, A, means):
    likelihoods = np.array([hmm_emission_likelihood(obs, m) for m in means])
    best_state = np.argmax(likelihoods)
    return states[best_state], likelihoods


# ============================================================
# 7. EXECUTAR PROJETO COMPLETO (COM SPIKE & FROZEN NO HMM)
# ============================================================

if __name__ == "__main__":
    Y, SlackBus, I = load_network()

    # ============================
    # PASSO 1 — BASELINE SE
    # ============================
    print("\n=== PASSO 1 — BASELINE SE ===")
    v0, z0, Hx, x0, r0 = compute_baseline(Y, SlackBus, I)
    thr_SE = 30 * np.median(np.abs(r0))
    print("Threshold SE (heurístico):", thr_SE)

    # ============================
    # PASSO 2 — FALHAS SE
    # ============================

    print("\n=== PASSO 2 — LINE OUTAGE 1–2 ===")
    Y_fault = simulate_line_outage(Y, (0,1))
    x_fault, r_fault = run_SE(Y_fault, SlackBus, I, z0)

    print("\n=== PASSO 2 — METER FAILURE (I12 corrompido) ===")
    z_fault = simulate_meter_failure(z0, 0, corruption=20)
    x_fault2, r_fault2 = run_SE(Y, SlackBus, I, z_fault)

    # ============================
    # PASSO 3 — AUTO‑REGRESSÃO (BASELINE)
    # ============================

    print("\n=== PASSO 3 — AUTO‑REGRESSÃO ===")
    I123, V3 = generate_time_series(Y, SlackBus, I)

    # AR para I123
    a_I, b_I = fit_AR1(I123)
    r_I, _ = compute_residuals(I123, a_I, b_I)
    anomalies_I, thr_I = detect_anomalies(r_I)
    frozen_I = detect_frozen_meter(I123)

    # AR para V3
    a_V, b_V = fit_AR1(V3)
    r_V, _ = compute_residuals(V3, a_V, b_V)
    anomalies_V, thr_V = detect_anomalies(r_V)
    frozen_V = detect_frozen_meter(V3)

    # ============================
    # PASSO 4 — FALHAS ADICIONAIS (SE + AR)
    # ============================

    print("\n=== PASSO 4 — ABNORMAL LOAD (bus 2 x 2.0) ===")
    I_abnormal = simulate_abnormal_load(I, bus=1, scale=2.0)
    v_ab, z_ab, Hx_ab, x_ab, r_ab = compute_baseline(Y, SlackBus, I_abnormal)

    # gerar série temporal para abnormal load
    I123_ab, V3_ab = generate_time_series(Y, SlackBus, I_abnormal)
    a_I_ab, b_I_ab = fit_AR1(I123_ab)
    r_I_ab, _ = compute_residuals(I123_ab, a_I_ab, b_I_ab)
    a_V_ab, b_V_ab = fit_AR1(V3_ab)
    r_V_ab, _ = compute_residuals(V3_ab, a_V_ab, b_V_ab)

    print("\n=== PASSO 4 — POWER FLOW FAULT (factor 1.5) ===")
    z_pf = simulate_power_flow_fault(Y, SlackBus, I, factor=1.5)
    x_pf, r_pf = run_SE(Y, SlackBus, I, z_pf)

    # para power_flow_fault, usamos AR baseline (podes mudar se quiseres outra simulação)

    # ============================
    # PASSO 5 — AR EM FALHAS (SPIKE & FROZEN)
    # ============================

    print("\n=== PASSO 5 — AR EM FALHAS (SPIKE & FROZEN) ===")

    # Spike em I123 e V3
    I123_spike = simulate_AR_spike(I123, t_spike=100, factor=3.0)
    r_I_spike, _ = compute_residuals(I123_spike, a_I, b_I)

    V3_spike = simulate_AR_spike(V3, t_spike=100, factor=3.0)
    r_V_spike, _ = compute_residuals(V3_spike, a_V, b_V)

    # Frozen em I123 e V3
    I123_frozen = simulate_AR_frozen(I123, t_start=100)
    r_I_frozen, _ = compute_residuals(I123_frozen, a_I, b_I)

    V3_frozen = simulate_AR_frozen(V3, t_start=100)
    r_V_frozen, _ = compute_residuals(V3_frozen, a_V, b_V)

    # ============================
    # VETORES DE OBSERVAÇÃO PARA HMM (AGORA COM SPIKE & FROZEN)
    # ============================

    observations = {}
    observations["normal"] = build_observation_vector(r0, r_I, r_V)
    observations["line_outage"] = build_observation_vector(r_fault, r_I, r_V)
    observations["meter_failure"] = build_observation_vector(r_fault2, r_I, r_V)
    observations["abnormal_load"] = build_observation_vector(r_ab, r_I_ab, r_V_ab)
    observations["power_flow_fault"] = build_observation_vector(r_pf, r_I, r_V)
    observations["spike"] = build_observation_vector(r0, r_I_spike, r_V_spike)
    observations["frozen_meter"] = build_observation_vector(r0, r_I_frozen, r_V_frozen)

    print("\n=== VETORES DE OBSERVAÇÃO (para HMM) ===")
    for k, v in observations.items():
        print(k, ":", v)

    # ============================
    # PASSO 6 — HMM PARA CLASSIFICAÇÃO
    # ============================

    print("\n=== PASSO 6 — HMM PARA CLASSIFICAÇÃO DE FALHAS ===")

    states, A, means = hmm_train(observations)

    for scenario, obs in observations.items():
        state_clean, _ = hmm_classify(obs, states, A, means)

        noise_level = 0.2
        noise = np.random.randn(*obs.shape) * noise_level
        obs_noisy = obs + noise
        state_noisy, _ = hmm_classify(obs_noisy, states, A, means)

        print(f"\nScenario: {scenario}")
        print("Observation (clean):", obs)
        print("Class (clean):", state_clean)
        print("Observation (noisy):", obs_noisy)
        print("Class (noisy):", state_noisy)

    # import matplotlib.pyplot as plt
    # print("\n================ RESULTADOS PARA O SLIDE ================\n")

    # # ============================================================
    # # 1) SE — BASELINE
    # # ============================================================
    # print("SE — Baseline")
    # print("Residuals (magnitude):", np.abs(r0))
    # print("Threshold SE:", thr_SE)

    # plt.figure(figsize=(8,4))
    # plt.stem(np.abs(r0))
    # plt.axhline(thr_SE, color='red', linestyle='--', label='Threshold')
    # plt.title("SE Residuals — Baseline")
    # plt.xlabel("Measurement index")
    # plt.ylabel("|residual|")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()


    # # ============================================================
    # # 2) SE — LINE OUTAGE
    # # ============================================================
    # print("\nSE — Line Outage (1–2)")
    # print("Residuals (magnitude):", np.abs(r_fault))

    # plt.figure(figsize=(8,4))
    # plt.stem(np.abs(r_fault))
    # plt.title("SE Residuals — Line Outage")
    # plt.xlabel("Measurement index")
    # plt.ylabel("|residual|")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()


    # # ============================================================
    # # 3) SE — METER FAILURE
    # # ============================================================
    # print("\nSE — Meter Failure (I12 corrupted)")
    # print("Residuals (magnitude):", np.abs(r_fault2))

    # plt.figure(figsize=(8,4))
    # plt.stem(np.abs(r_fault2))
    # plt.title("SE Residuals — Meter Failure")
    # plt.xlabel("Measurement index")
    # plt.ylabel("|residual|")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()


    # # ============================================================
    # # 4) AR — BASELINE, SPIKE, FROZEN
    # # ============================================================

    # # Garantir que as variáveis existem
    # anomalies_I_spike, _ = detect_anomalies(r_I_spike)
    # anomalies_I_frozen, _ = detect_anomalies(r_I_frozen)

    # print("\nAR — Baseline I123")
    # print("Spike anomalies:", len(anomalies_I_spike))
    # print("Frozen anomalies:", len(anomalies_I_frozen))

    # # Série temporal baseline
    # plt.figure(figsize=(10,4))
    # plt.plot(I123, label="I123 (time series)")
    # plt.title("Time Series — I123 (Baseline)")
    # plt.xlabel("Time")
    # plt.ylabel("Current")
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    # # Resíduos baseline
    # plt.figure(figsize=(10,4))
    # plt.plot(r_I, label="AR(1) Residuals")
    # plt.axhline(thr_I, color='red', linestyle='--', label='AR Threshold')
    # plt.title("AR(1) Residuals — I123 (Baseline)")
    # plt.xlabel("Time")
    # plt.ylabel("Residual")
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    # # Spike
    # plt.figure(figsize=(10,4))
    # plt.plot(r_I_spike, label="Residuals (Spike)")
    # plt.title("AR(1) Residuals — I123 Spike")
    # plt.xlabel("Time")
    # plt.ylabel("Residual")
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    # # Frozen
    # plt.figure(figsize=(10,4))
    # plt.plot(r_I_frozen, label="Residuals (Frozen)")
    # plt.title("AR(1) Residuals — I123 Frozen")
    # plt.xlabel("Time")
    # plt.ylabel("Residual")
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout()
    # plt.show()


    # # ============================================================
    # # 5) HMM — OBSERVATION VECTORS
    # # ============================================================
    # print("\nHMM — Observation Vectors:")
    # for k, v in observations.items():
    #     print(f"{k}: {v}")

    # states_list = list(observations.keys())
    # obs_matrix = np.array([observations[s] for s in states_list])

    # plt.figure(figsize=(10,5))
    # x = np.arange(len(states_list))
    # width = 0.25

    # plt.bar(x - width, obs_matrix[:,0], width, label='max|r_SE|')
    # plt.bar(x,         obs_matrix[:,1], width, label='max|r_AR_I|')
    # plt.bar(x + width, obs_matrix[:,2], width, label='max|r_AR_V|')

    # plt.xticks(x, states_list, rotation=20)
    # plt.ylabel("Magnitude")
    # plt.title("Observation Vectors per Scenario (HMM)")
    # plt.legend()
    # plt.grid(axis='y', linestyle='--', alpha=0.6)
    # plt.tight_layout()
    # plt.show()


    # # ============================================================
    # # 6) HMM — CLASSIFICATION (CLEAN + NOISY)
    # # ============================================================
    # print("\nHMM — Classification Results")

    # for scenario, obs in observations.items():
    #     state_clean, _ = hmm_classify(obs, states, A, means)

    #     noise = np.random.randn(*obs.shape) * 0.05
    #     obs_noisy = obs + noise
    #     state_noisy, _ = hmm_classify(obs_noisy, states, A, means)

    #     print(f"\nScenario: {scenario}")
    #     print("  Clean observation:", obs)
    #     print("  Classified as:", state_clean)
    #     print("  Noisy observation:", obs_noisy)
    #     print("  Classified as:", state_noisy)

    # print("\n================ FIM DOS RESULTADOS ================\n")
