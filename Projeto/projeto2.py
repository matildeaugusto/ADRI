import numpy as np

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

def pf3ph_neutral(topo, z_phase, z_neutral, si, vr, el, ni, al):

    p = topo[:,0]
    f = topo[:,1]
    w = len(p)+1

    V = np.zeros((4, w), dtype=complex)
    V[:,0] = np.array([vr, vr*al, vr*al**2, 0])

    I = np.zeros((4, w), dtype=complex)

    Z = np.diag([z_phase[0], z_phase[1], z_phase[2], z_neutral])

    Z_ground = 0.001  # ligação neutro-terra no bus 1

    for _ in range(ni):

        I[:] = 0

        # BACKWARD
        for k in range(w-1, 0, -1):
            n = f[k-1]-1
            m = p[k-1]-1

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
        for k in range(w-1):
            n = f[k]-1
            m = p[k]-1
            V[:,n] = V[:,m] - Z @ I[:,n]

        # ligação neutro-terra no bus 1
        V[3,0] = - Z_ground * I[3,0]

    return V

# ============================================================
# PASSO 2 — OPERAÇÃO NORMAL
# ============================================================

def build_si_from_row(row, nBUS):
    si = np.zeros((3, nBUS))
    si[2,0] = row[2]  # cliente 3
    si[2,1] = row[1]  # cliente 2
    si[1,2] = row[0]  # cliente 1
    si[2,2] = row[3]  # cliente 4
    return si

def simulate_normal(s, Zn=Zn_normal):

    m = len(s)
    Volt = np.zeros((m, 3), dtype=complex)

    for i in range(m):
        si = build_si_from_row(s[i,:], nBUS)
        V = pf3ph_neutral(topo, z_phase, Zn, si, vr, el, ni, al)
        Volt[i,:] = V[0:3, -1] - V[3,-1]

    return Volt

baseline = simulate_normal(s)
print("\n=== PASSO 2 — OPERAÇÃO NORMAL ===")
print(baseline)

# ============================================================
# PASSO 3 — FALHA NO NEUTRO
# ============================================================

def simulate_fault(s, Zn_fault=1e6):

    m = len(s)
    Volt_fault = np.zeros((m, 3), dtype=complex)

    for i in range(m):
        si = build_si_from_row(s[i,:], nBUS)
        V = pf3ph_neutral(topo, z_phase, Zn_fault, si, vr, el, ni, al)
        Volt_fault[i,:] = V[0:3, -1] - V[3,-1]

    return Volt_fault

fault = simulate_fault(s)
print("\n=== PASSO 3 — FALHA NO NEUTRO ===")
print(fault)

# ============================================================
# PASSO 4 — INDICADORES DE FALHA
# ============================================================

def compute_indicators(Volt_normal, Volt_fault):

    Vn = np.abs(Volt_normal)
    Vf = np.abs(Volt_fault)

    ind_unbalance_normal = np.max(Vn, axis=1) - np.min(Vn, axis=1)
    ind_unbalance_fault  = np.max(Vf, axis=1) - np.min(Vf, axis=1)

    thd_normal = np.std(Vn, axis=1) / np.mean(Vn, axis=1)
    thd_fault  = np.std(Vf, axis=1) / np.mean(Vf, axis=1)

    In_proxy_normal = ind_unbalance_normal
    In_proxy_fault  = ind_unbalance_fault

    return {
        "unbalance_normal": ind_unbalance_normal,
        "unbalance_fault": ind_unbalance_fault,
        "thd_normal": thd_normal,
        "thd_fault": thd_fault,
        "In_normal": In_proxy_normal,
        "In_fault": In_proxy_fault
    }

ind = compute_indicators(baseline, fault)

print("\n=== PASSO 4 — INDICADORES DE FALHA ===")
print("Desequilíbrio normal:", ind["unbalance_normal"])
print("Desequilíbrio falha :", ind["unbalance_fault"])

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

            Volt_norm = simulate_normal(s_scaled, Zn=Zn)
            Volt_fault = simulate_fault(s_scaled, Zn_fault=1e6)

            ind_loc = compute_indicators(Volt_norm, Volt_fault)

            mean_unb_norm  = np.mean(ind_loc["unbalance_normal"])
            mean_unb_fault = np.mean(ind_loc["unbalance_fault"])

            results.append({
                "load_factor": lf,
                "Zn": Zn,
                "mean_unbalance_normal": mean_unb_norm,
                "mean_unbalance_fault": mean_unb_fault
            })

    return results

load_factors = [0.5, 1.0, 1.5, 2.0]
Zn_values = [0.001, 0.01, 0.1]

validation_results = sweep_loads_and_Zn(s, load_factors, Zn_values)

print("\n=== PASSO 5 — VALIDAÇÃO DO MÉTODO ===")
for r in validation_results:
    print(f"Load x{r['load_factor']:.1f}, Zn={r['Zn']}: "
          f"Unb_normal_mean={r['mean_unbalance_normal']:.4f}, "
          f"Unb_fault_mean={r['mean_unbalance_fault']:.4f}")

# ============================================================
# PASSO 6 — INDICADOR BINÁRIO DE FALHA
# ============================================================

def binary_fault_indicator(Volt_normal, Volt_fault):

    Vn = np.abs(Volt_normal)
    Vf = np.abs(Volt_fault)

    unb_norm = np.max(Vn, axis=1) - np.min(Vn, axis=1)
    unb_fault = np.max(Vf, axis=1) - np.min(Vf, axis=1)

    thr = np.mean(unb_norm) + 3*np.std(unb_norm)

    bin_norm = (unb_norm > thr).astype(int)
    bin_fault = (unb_fault > thr).astype(int)

    global_norm = int(np.any(bin_norm == 1))
    global_fault = int(np.any(bin_fault == 1))

    return {
        "threshold": thr,
        "binary_normal": bin_norm,
        "binary_fault": bin_fault,
        "global_normal": global_norm,
        "global_fault": global_fault
    }

bin_ind = binary_fault_indicator(baseline, fault)

print("\n=== PASSO 6 — INDICADOR BINÁRIO ===")
print(f"Threshold automático: {bin_ind['threshold']:.4f}")
print("Normal:", bin_ind["binary_normal"])
print("Falha :", bin_ind["binary_fault"])
print("Resumo → normal detetado como falha:", bin_ind["global_normal"])
print("Resumo → falha detetada:", bin_ind["global_fault"])
