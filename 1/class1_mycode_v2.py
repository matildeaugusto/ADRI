import numpy as np
from scipy.optimize import nnls
import matplotlib.pyplot as plt

# ============================================================
# PARÂMETROS
# ============================================================

nc = 4  # número de consumidores
ts = 60
te = 72
noise = 0.02
np.random.seed(42)  # para reprodutibilidade

# ============================================================
# DADOS SINTÉTICOS (sem dependência do Excel)
# ============================================================

# Criar 4 consumidores com perfis distintos
n_periods = te - ts + 1
X_synthetic = np.array([
    [1.0, 1.0, 0.5, 0.8],  # período 0: C1=1, C2=1 (idênticos!), C3=0.5, C4=0.8
    [1.2, 1.2, 0.6, 0.9],  # período 1
    [0.9, 0.9, 0.7, 1.0],  # período 2
    [1.1, 1.1, 0.6, 0.7],
    [1.0, 1.0, 0.8, 0.9],
    [1.3, 1.3, 0.5, 1.1],
    [0.8, 0.8, 0.9, 0.6],
    [1.2, 1.2, 0.7, 0.8],
    [1.1, 1.1, 0.6, 1.0],
    [0.9, 0.9, 0.8, 0.9],
    [1.0, 1.0, 0.7, 0.7],
    [1.1, 1.1, 0.6, 0.8],
    [1.2, 1.2, 0.5, 1.0],
])

print(f"Matriz X (poder dos consumidores): shape={X_synthetic.shape}")
print(X_synthetic)

# Fases verdadeiras: C1→Fase1, C2→Fase1, C3→Fase2, C4→Fase3
true_phases = np.array([1, 1, 2, 3])
true_phase_idx = true_phases - 1

print(f"\nFases verdadeiras: {true_phases}")

# ============================================================
# CONSTRUIR Y VERDADEIRO (soma de consumo por fase)
# ============================================================

Y_true = np.zeros((n_periods, 3))
for f in range(3):
    Y_true[:, f] = X_synthetic[:, true_phase_idx == f].sum(axis=1)

# Adicionar ruído
Y_noisy = Y_true + noise * np.random.randn(*Y_true.shape)

print(f"\nY_true (soma por fase):\n{Y_true}")
print(f"\nY_noisy:\n{Y_noisy}")

# ============================================================
# DETECTAR CONSUMIDORES IDÊNTICOS
# ============================================================

def find_identical_consumers(X, tol=1e-10):
    n = X.shape[1]
    identical_groups = []
    used = set()
    
    for i in range(n):
        if i in used:
            continue
        group = [i]
        for j in range(i + 1, n):
            if j in used:
                continue
            if np.allclose(X[:, i], X[:, j], atol=tol):
                group.append(j)
                used.add(j)
        
        if len(group) > 1:
            identical_groups.append(group)
            used.add(i)
    
    return identical_groups

identical_groups = find_identical_consumers(X_synthetic)
print(f"\nConsumidores idênticos detectados: {identical_groups}")

# ============================================================
# RESOLVER COM NNLS (SEM QUALQUER AJUSTE)
# ============================================================

print("\n" + "="*60)
print("MÉTODO 1: NNLS PURO (SEM AJUSTES)")
print("="*60)

B_nnls = np.zeros((nc, 3))
for f in range(3):
    b_f, residual = nnls(X_synthetic, Y_noisy[:, f])
    B_nnls[:, f] = b_f

print("\nB estimado (NNLS puro):")
print(B_nnls)

# Arredondar para fases
estimated_phases_nnls = np.argmax(B_nnls, axis=1) + 1
print(f"\nFases estimadas: {estimated_phases_nnls}")
print(f"Fases verdadeiras: {true_phases}")
accuracy_nnls = np.mean(estimated_phases_nnls == true_phases)
print(f"Accuracy (NNLS): {accuracy_nnls*100:.1f}%")

# ============================================================
# RESOLVER COM RESTRIÇÃO DE QUEBRA DE SIMETRIA
# ============================================================

print("\n" + "="*60)
print("MÉTODO 2: COM QUEBRA DE SIMETRIA (distribuir por fases diferentes)")
print("="*60)

def estimate_B_with_symmetry_breaking(X, Y, identical_groups):
    n_consumers = X.shape[1]
    B = np.zeros((n_consumers, 3))
    
    # Resolver NNLS standard
    for f in range(3):
        b_f, _ = nnls(X, Y[:, f])
        B[:, f] = b_f
    
    # Quebra de simetria
    for group in identical_groups:
        if len(group) <= 3:
            B_group = B[group, :].copy()
            sums_per_phase = B_group.sum(axis=0)
            
            # Distribuir cíclicamente por fases diferentes
            for i, idx in enumerate(group):
                phase_assignment = i % 3
                B[idx, :] = 0
                B[idx, phase_assignment] = sums_per_phase[phase_assignment] / len(group)
    
    return B

B_break = estimate_B_with_symmetry_breaking(X_synthetic, Y_noisy, identical_groups)
print("\nB estimado (com quebra de simetria):")
print(B_break)

estimated_phases_break = np.argmax(B_break, axis=1) + 1
print(f"\nFases estimadas: {estimated_phases_break}")
print(f"Fases verdadeiras: {true_phases}")
accuracy_break = np.mean(estimated_phases_break == true_phases)
print(f"Accuracy (com quebra): {accuracy_break*100:.1f}%")

# ============================================================
# RESUMO
# ============================================================

print("\n" + "="*60)
print("RESUMO")
print("="*60)
print(f"Consumidores idênticos: {identical_groups}")
print(f"\nNNLS puro:")
print(f"  Fases: {estimated_phases_nnls}")
print(f"  Accuracy: {accuracy_nnls*100:.1f}%")
print(f"\nCom quebra de simetria:")
print(f"  Fases: {estimated_phases_break}")
print(f"  Accuracy: {accuracy_break*100:.1f}%")
