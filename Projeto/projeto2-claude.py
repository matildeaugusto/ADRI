import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# DADOS DO LAB (dataset real)
# ============================================================

s = np.array([
 [0.0450, 0.0150, 0.0470, 0.0330],
 [0.0250, 0.0150, 0.2480, 0.0330],
 [0.0970, 0.0250, 0.3940, 0.0330],
 [0.0700, 0.0490, 0.0200, 0.4850],
 [0.1250, 0.0460, 0.0160, 0.1430],
 [0.2900, 0.0270, 0.0160, 0.0470],
 [0.2590, 0.0150, 0.0170, 0.0200],
 [0.2590, 0.0160, 0.0280, 0.0160],
 [0.4420, 0.0160, 0.0500, 0.0170],
 [0.2010, 0.0230, 0.0460, 0.0160],
 [0.2060, 0.0490, 0.0220, 0.0240],
 [0.1300, 0.0470, 0.0160, 0.0490],
 [0.0460, 0.0260, 0.0170, 0.0480]
])

# ============================================================
# PARÂMETROS
# ============================================================

vr = 1.0
el = 1
ni = 20
netFactor = 0.25
al = np.exp(-1j * 2*np.pi/3)

# ============================================================
# PASSO 1 — REDE RADIAL
# ============================================================

topo = np.array([
    [1, 2],
    [2, 3],
    [3, 4]
])

nBUS = 4

z_base = np.array([
    0.1 + 0.05j,
    0.15 + 0.07j,
    0.2 + 0.10j
])

z_phase = z_base * netFactor
Zn_normal = 0.01

print("Rede radial criada.")

# ============================================================
# PF TRIFÁSICO COM NEUTRO EXPLÍCITO + TERRA
# ============================================================

# FIX 1: Added z_ground as an explicit parameter (was hardcoded at 0.001 inside).
#         This allows it to be varied in sensitivity tests.
# FIX 2: Z matrix is now built per-branch inside the forward sweep.
#         Original code built one fixed diagonal Z = diag([z_phase[0], z_phase[1],
#         z_phase[2], z_neutral]), which incorrectly assigned a different impedance
#         to each phase (conflating branch index with phase index). All three phases
#         of a given branch share the same conductor impedance z_phase[k].
# FIX 3: Added a convergence tolerance check (eps) so the loop exits early once
#         voltages stabilise, rather than always running all ni iterations.

def pf3ph_neutral(topo, z_phase, z_neutral, si, vr, el, ni, al, z_ground=0.001):
    # FIX 1: z_ground is now a parameter, not hardcoded.

    p = topo[:,0]
    f = topo[:,1]
    w = len(p) + 1

    V = np.zeros((4, w), dtype=complex)
    V[:,0] = np.array([vr, vr*al, vr*al**2, 0])

    I = np.zeros((4, w), dtype=complex)

    # FIX 3: convergence tolerance
    tol = 1e-6

    for _ in range(ni):

        I[:] = 0
        V_prev = V.copy()

        # BACKWARD
        for k in range(w-1, 0, -1):
            n = f[k-1] - 1
            m = p[k-1] - 1

            Va = V[0,n] - V[3,n]
            Vb = V[1,n] - V[3,n]
            Vc = V[2,n] - V[3,n]

            Vph = np.array([Va, Vb, Vc])
            Vph[Vph == 0] = 1e-12

            I_load = np.conj(si[:,n] * np.abs(Vph)**el / Vph)
            I_n = -(I_load.sum())

            I[:,n] += np.array([I_load[0], I_load[1], I_load[2], I_n])
            I[:,m] += I[:,n]

        # FORWARD
        # FIX 2: build a fresh per-branch Z at each branch k, so all three
        #         phases of branch k use z_phase[k] (not three different values).
        for k in range(w-1):
            n = f[k] - 1
            m = p[k] - 1
            Zk = np.diag([z_phase[k], z_phase[k], z_phase[k], z_neutral])  # FIX 2
            V[:,n] = V[:,m] - Zk @ I[:,n]

        # Neutral-to-ground bond at bus 0
        V[3,0] = -z_ground * I[3,0]  # FIX 1: uses parameter

        # FIX 3: check convergence
        if np.max(np.abs(V - V_prev)) < tol:
            break

    return V

# ============================================================
# PASSO 2 — OPERAÇÃO NORMAL
# ============================================================

# FIX 4: Consumer 3 was placed at bus index 0 (the source bus), so its current
#         was injected but never propagated backward — effectively lost.
#         Corrected mapping (matching the class notebook reference):
#           Consumer 3 (row[2]) → Phase A, bus 3 (index 2)
#           Consumer 2 (row[1]) → Phase B, bus 3 (index 2)
#           Consumer 1 (row[0]) → Phase C, bus 2 (index 1)
#           Consumer 4 (row[3]) → Phase C, bus 4 (index 3)
#
# Original (wrong):
#   si[2,0] = row[2]  # consumer 3 placed at source bus!
#   si[2,1] = row[1]
#   si[1,2] = row[0]
#   si[2,2] = row[3]

def build_si_from_row(row, nBUS):
    si = np.zeros((3, nBUS))
    si[0,2] = row[2]  # FIX 4: consumer 3 → Phase A, bus 3
    si[1,2] = row[1]  # FIX 4: consumer 2 → Phase B, bus 3
    si[2,1] = row[0]  # FIX 4: consumer 1 → Phase C, bus 2
    si[2,3] = row[3]  # FIX 4: consumer 4 → Phase C, bus 4
    return si

def simulate_normal(s, Zn=Zn_normal):

    m = len(s)
    Volt = np.zeros((m, 3), dtype=complex)

    for i in range(m):
        si = build_si_from_row(s[i,:], nBUS)
        V = pf3ph_neutral(topo, z_phase, Zn, si, vr, el, ni, al)
        Volt[i,:] = V[0:3, -1] - V[3,-1]

    return Volt

# ============================================================
# PASSO 3 — FALHA NO NEUTRO
# ============================================================

def simulate_fault(s, Zn_fault=1e6):
    # Note: z_neutral=1e6 models an open neutral wire.
    # The phase conductors and z_ground bond are unchanged.

    m = len(s)
    Volt_fault = np.zeros((m, 3), dtype=complex)

    for i in range(m):
        si = build_si_from_row(s[i,:], nBUS)
        V = pf3ph_neutral(topo, z_phase, Zn_fault, si, vr, el, ni, al)
        Volt_fault[i,:] = V[0:3, -1] - V[3,-1]

    return Volt_fault

# ============================================================
# PASSO 4 — INDICADORES DE FALHA
# ============================================================

# FIX 5: Renamed 'thd_normal/fault' → 'vuf_normal/fault' (Voltage Unbalance Factor).
#         std(|V|)/mean(|V|) measures voltage unbalance, not harmonic distortion (THD).
#         Similarly renamed 'In_proxy' → 'unbalance_proxy' for clarity.

def compute_indicators(Volt_normal, Volt_fault):

    Vn = np.abs(Volt_normal)
    Vf = np.abs(Volt_fault)

    ind_unbalance_normal = np.max(Vn, axis=1) - np.min(Vn, axis=1)
    ind_unbalance_fault  = np.max(Vf, axis=1) - np.min(Vf, axis=1)

    vuf_normal = np.std(Vn, axis=1) / np.mean(Vn, axis=1)   # FIX 5: was thd_normal
    vuf_fault  = np.std(Vf, axis=1) / np.mean(Vf, axis=1)   # FIX 5: was thd_fault

    return {
        "unbalance_normal":       ind_unbalance_normal,
        "unbalance_fault":        ind_unbalance_fault,
        "vuf_normal":             vuf_normal,               # FIX 5
        "vuf_fault":              vuf_fault,                # FIX 5
        "unbalance_proxy_normal": ind_unbalance_normal,     # FIX 5: was In_proxy_normal
        "unbalance_proxy_fault":  ind_unbalance_fault,      # FIX 5: was In_proxy_fault
    }

# ============================================================
# PASSO 5 — VALIDAR MÉTODO (SENSIBILIDADE E ROBUSTEZ)
# ============================================================

def scale_loads(s, factor):
    return s * factor

def sweep_loads_and_Zn(s, load_factors, Zn_values):

    results = []

    for lf in load_factors:
        s_scaled = scale_loads(s, lf)

        for Zn in Zn_values:

            Volt_norm  = simulate_normal(s_scaled, Zn=Zn)
            Volt_fault = simulate_fault(s_scaled, Zn_fault=1e6)

            ind_loc = compute_indicators(Volt_norm, Volt_fault)

            mean_unb_norm  = np.mean(ind_loc["unbalance_normal"])
            mean_unb_fault = np.mean(ind_loc["unbalance_fault"])

            results.append({
                "load_factor":           lf,
                "Zn":                    Zn,
                "mean_unbalance_normal": mean_unb_norm,
                "mean_unbalance_fault":  mean_unb_fault
            })

    return results

load_factors = [0.5, 1.0, 1.5, 2.0]
Zn_values    = [0.001, 0.01, 0.1]

# ============================================================
# PASSO 6 — INDICADOR BINÁRIO DE FALHA
# ============================================================

def binary_fault_indicator(Volt_normal, Volt_fault):

    Vn = np.abs(Volt_normal)
    Vf = np.abs(Volt_fault)

    unb_norm  = np.max(Vn, axis=1) - np.min(Vn, axis=1)
    unb_fault = np.max(Vf, axis=1) - np.min(Vf, axis=1)

    thr = np.mean(unb_norm) + 3*np.std(unb_norm)

    bin_norm  = (unb_norm  > thr).astype(int)
    bin_fault = (unb_fault > thr).astype(int)

    global_norm  = int(np.any(bin_norm  == 1))
    global_fault = int(np.any(bin_fault == 1))

    return {
        "threshold":     thr,
        "binary_normal": bin_norm,
        "binary_fault":  bin_fault,
        "global_normal": global_norm,
        "global_fault":  global_fault
    }

# ============================================================
# RUÍDO — funções auxiliares
# ============================================================

# FIX 6: add_noise now clips to zero so voltage magnitudes cannot go negative.
def add_noise(V, sigma):
    noise = sigma * np.random.randn(*V.shape)
    return np.maximum(V + noise, 0.0)   # FIX 6: was just V + noise

def noise_sensitivity_test(s, noise_levels, N=50):

    results = []

    for sigma in noise_levels:

        correct_normal = 0
        correct_fault  = 0

        for _ in range(N):

            Vn = simulate_normal(s)
            Vf = simulate_fault(s)

            Vn_noisy = add_noise(np.abs(Vn), sigma)
            Vf_noisy = add_noise(np.abs(Vf), sigma)

            ind_n = np.max(Vn_noisy, axis=1) - np.min(Vn_noisy, axis=1)
            ind_f = np.max(Vf_noisy, axis=1) - np.min(Vf_noisy, axis=1)

            thr = np.mean(ind_n) + 3*np.std(ind_n)

            if np.all(ind_n < thr):
                correct_normal += 1
            if np.all(ind_f > thr):
                correct_fault += 1

        results.append({
            "sigma":           sigma,
            "normal_accuracy": correct_normal / N,
            "fault_accuracy":  correct_fault  / N
        })

    return results

# ============================================================
# PONTO DE ENTRADA — toda a simulação e todos os gráficos aqui
# ============================================================

# FIX 7: Removed all plt.show() calls that were at module level (outside __main__).
#         They caused every plot to appear twice when running the script directly,
#         and once extra when importing the module. All plots are now inside __main__.

if __name__ == "__main__":

    # ----------------------------------------------------------
    # Passo 2 — Operação normal
    # ----------------------------------------------------------
    print("\n=== PASSO 2 — OPERAÇÃO NORMAL ===")
    baseline = simulate_normal(s)
    print(baseline)

    # ----------------------------------------------------------
    # Passo 3 — Falha no neutro
    # ----------------------------------------------------------
    print("\n=== PASSO 3 — FALHA NO NEUTRO ===")
    fault = simulate_fault(s)
    print(fault)

    # ----------------------------------------------------------
    # Passo 4 — Indicadores
    # ----------------------------------------------------------
    ind = compute_indicators(baseline, fault)

    print("\n=== PASSO 4 — INDICADORES DE FALHA ===")
    print("Desequilíbrio normal:", ind["unbalance_normal"])
    print("Desequilíbrio falha :", ind["unbalance_fault"])

    # ----------------------------------------------------------
    # Passo 5 — Validação paramétrica
    # ----------------------------------------------------------
    validation_results = sweep_loads_and_Zn(s, load_factors, Zn_values)

    print("\n=== PASSO 5 — VALIDAÇÃO DO MÉTODO ===")
    for r in validation_results:
        print(f"Load x{r['load_factor']:.1f}, Zn={r['Zn']}: "
              f"Unb_normal_mean={r['mean_unbalance_normal']:.4f}, "
              f"Unb_fault_mean={r['mean_unbalance_fault']:.4f}")

    # ----------------------------------------------------------
    # Passo 6 — Indicador binário
    # ----------------------------------------------------------
    bin_ind = binary_fault_indicator(baseline, fault)

    print("\n=== PASSO 6 — INDICADOR BINÁRIO ===")
    print(f"Threshold automático: {bin_ind['threshold']:.4f}")
    print("Normal:", bin_ind["binary_normal"])
    print("Falha :", bin_ind["binary_fault"])
    print("Resumo → normal detetado como falha:", bin_ind["global_normal"])
    print("Resumo → falha detetada:",             bin_ind["global_fault"])

    # ==========================================================
    # GRÁFICO 1 — Tensões fase-neutro (Normal vs Falha)
    # ==========================================================

    idx = 0
    Vn_bar = np.abs(baseline[idx, :])
    Vf_bar = np.abs(fault[idx, :])

    plt.figure(figsize=(6,4))
    x = np.arange(3)
    w = 0.35
    plt.bar(x - w/2, Vn_bar, width=w, alpha=0.8, label="Normal")
    plt.bar(x + w/2, Vf_bar, width=w, alpha=0.8, label="Falha")
    plt.xticks(x, ["Va", "Vb", "Vc"])
    plt.title("Tensões fase-neutro — Normal vs Falha")
    plt.ylabel("Magnitude [pu]")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ==========================================================
    # GRÁFICO 2 — Desequilíbrio de tensão (Normal vs Falha + Threshold)
    # ==========================================================

    unb_norm  = ind["unbalance_normal"]
    unb_fault = ind["unbalance_fault"]
    thr       = bin_ind["threshold"]

    plt.figure(figsize=(6,4))
    plt.plot(unb_norm,  'o-', linewidth=2, markersize=6, label="Normal")
    plt.plot(unb_fault, 'o-', linewidth=2, markersize=6, label="Fault")
    plt.axhline(thr, color='red', linestyle='--', linewidth=2,
                label=f"Threshold = {thr:.3f}")
    plt.title("Voltage Unbalance — Normal vs Fault")
    plt.xlabel("Sample")
    plt.ylabel("Unbalance [pu]")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ==========================================================
    # HEATMAP — Desequilíbrio Normal (load × Zn)
    # ==========================================================

    unb_norm_matrix = np.zeros((len(load_factors), len(Zn_values)))
    k = 0
    for i in range(len(load_factors)):
        for j in range(len(Zn_values)):
            unb_norm_matrix[i, j] = validation_results[k]["mean_unbalance_normal"]
            k += 1

    plt.figure(figsize=(6,5))
    plt.imshow(unb_norm_matrix, cmap="viridis", aspect="auto")
    plt.colorbar(label="Desequilíbrio normal [pu]")
    plt.xticks(np.arange(len(Zn_values)), Zn_values)
    plt.yticks(np.arange(len(load_factors)), load_factors)
    plt.xlabel("Impedância do neutro Zn [pu]")
    plt.ylabel("Load factor")
    plt.title("Heatmap — Desequilíbrio Normal (load × Zn)")
    plt.tight_layout()
    plt.show()

    # ==========================================================
    # BARRAS — Desequilíbrio em Falha (load × Zn)
    # ==========================================================

    unb_fault_values = [r["mean_unbalance_fault"] for r in validation_results]

    plt.figure(figsize=(7,4))
    plt.bar(range(len(unb_fault_values)), unb_fault_values, color="orange")
    plt.axhline(np.mean(unb_fault_values), color="red", linestyle="--",
                label=f"Média ≈ {np.mean(unb_fault_values):.3f}")
    plt.title("Desequilíbrio em Falha — Sempre Elevado e Constante")
    plt.ylabel("Desequilíbrio [pu]")
    plt.xlabel("Cenário (load × Zn)")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # ==========================================================
    # SENSIBILIDADE AO RUÍDO
    # ==========================================================

    noise_levels = [0, 0.05, 0.1, 0.15, 0.2]
    sens = noise_sensitivity_test(s, noise_levels)

    plt.figure(figsize=(6,4))
    plt.plot(noise_levels, [r["normal_accuracy"] for r in sens],
             'o-', label="Normal correctly classified")
    plt.plot(noise_levels, [r["fault_accuracy"]  for r in sens],
             'o-', label="Fault correctly detected")
    plt.xlabel("Noise level")
    plt.ylabel("Accuracy")
    plt.title("Sensitivity to Measurement Noise")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()