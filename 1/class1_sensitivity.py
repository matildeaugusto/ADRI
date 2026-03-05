import numpy as np
from scipy.optimize import nnls
import matplotlib.pyplot as plt

# ============================================================
# PARÂMETROS
# ============================================================

nc = 4
noise = 0.02
np.random.seed(42)

# Base: 4 consumidores, períodos 60-72 (13 períodos)
n_periods = 13

# ============================================================
# DADOS SINTÉTICOS: dois consumidores quase idênticos em FASES DIFERENTES
# ============================================================

# Consumidor 0: perfil de consumo "A"
# Consumidor 1: perfil de consumo "A" mas com pequena perturbação (variável!)
# Consumidor 2, 3: outros perfis

profile_A = np.array([1.0, 1.2, 0.9, 1.1, 1.0, 1.3, 0.8, 1.2, 1.1, 0.9, 1.0, 1.1, 1.2])
profile_B = np.array([0.5, 0.6, 0.7, 0.6, 0.8, 0.5, 0.9, 0.7, 0.6, 0.8, 0.7, 0.6, 0.5])
profile_C = np.array([0.8, 0.9, 1.0, 0.7, 0.9, 1.1, 0.6, 0.8, 1.0, 0.9, 0.7, 0.8, 1.0])

# Fases verdadeiras:
# Consumidor 0 → Fase 1
# Consumidor 1 → Fase 2  (DIFERENTE de 0!)
# Consumidor 2 → Fase 2
# Consumidor 3 → Fase 3
true_phases = np.array([1, 2, 2, 3])
true_phase_idx = true_phases - 1

print("="*70)
print("TESTE DE SENSIBILIDADE: Dois consumidores quase idênticos em fases diferentes")
print("="*70)
print(f"\nFases verdadeiras: {true_phases}")
print("Consumidor 0: Fase 1 (perfil A)")
print("Consumidor 1: Fase 2 (perfil A com perturbação)")
print("Consumidor 2: Fase 2 (perfil B)")
print("Consumidor 3: Fase 3 (perfil C)")

# ============================================================
# VARIAR EPSILON E MEDIR SENSIBILIDADE
# ============================================================

epsilons = np.array([0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2])
results = []

for eps in epsilons:
    print(f"\n" + "-"*70)
    print(f"EPSILON = {eps:.4f} (diferença relativa: {eps*100:.2f}%)")
    print("-"*70)
    
    # Construir X: consumidores 0 e 1 têm consumos muito parecidos
    X = np.column_stack([
        profile_A,                      # Consumidor 0 (Fase 1)
        profile_A * (1 + eps),          # Consumidor 1 (Fase 2), perturbado
        profile_B,                      # Consumidor 2 (Fase 2)
        profile_C                       # Consumidor 3 (Fase 3)
    ])
    
    # Y verdadeiro: soma de consumo por fase
    Y_true = np.zeros((n_periods, 3))
    for f in range(3):
        Y_true[:, f] = X[:, true_phase_idx == f].sum(axis=1)
    
    # Adicionar ruído
    Y_noisy = Y_true + noise * np.random.randn(*Y_true.shape)
    
    # Resolver com NNLS
    B = np.zeros((nc, 3))
    for f in range(3):
        b_f, _ = nnls(X, Y_noisy[:, f])
        B[:, f] = b_f
    
    # Arredondar para fases
    estimated_phases = np.argmax(B, axis=1) + 1
    
    # Calcular accuracy
    accuracy = np.mean(estimated_phases == true_phases)
    
    # Erro específico: consumidores 0 e 1 foram atribuídos às fases corretas?
    c0_correct = (estimated_phases[0] == true_phases[0])
    c1_correct = (estimated_phases[1] == true_phases[1])
    
    print(f"B estimado:\n{B}")
    print(f"\nFases verdadeiras: {true_phases}")
    print(f"Fases estimadas:   {estimated_phases}")
    print(f"\nConsumidor 0 (Fase 1): {'✓' if c0_correct else '✗'} - estimado como Fase {estimated_phases[0]}")
    print(f"Consumidor 1 (Fase 2): {'✓' if c1_correct else '✗'} - estimado como Fase {estimated_phases[1]}")
    print(f"Consumidor 2 (Fase 2): {'✓' if estimated_phases[2] == true_phases[2] else '✗'}")
    print(f"Consumidor 3 (Fase 3): {'✓' if estimated_phases[3] == true_phases[3] else '✗'}")
    print(f"\nAccuracy total: {accuracy*100:.1f}%")
    
    results.append({
        'epsilon': eps,
        'accuracy': accuracy,
        'c0_correct': c0_correct,
        'c1_correct': c1_correct,
        'B': B.copy(),
        'phases_est': estimated_phases.copy()
    })

# ============================================================
# RESUMO E GRÁFICO
# ============================================================

print("\n" + "="*70)
print("RESUMO DA SENSIBILIDADE")
print("="*70)

eps_vals = np.array([r['epsilon'] for r in results])
acc_vals = np.array([r['accuracy'] for r in results])
c0_correct = np.array([r['c0_correct'] for r in results])
c1_correct = np.array([r['c1_correct'] for r in results])

for i, r in enumerate(results):
    print(f"ε={r['epsilon']:.4f}: Accuracy={r['accuracy']*100:5.1f}% | C0={r['c0_correct']} | C1={r['c1_correct']}")

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Gráfico 1: Accuracy vs Epsilon
ax1.plot(eps_vals * 100, acc_vals * 100, 'o-', linewidth=2, markersize=8, color='blue')
ax1.axhline(y=100, color='green', linestyle='--', alpha=0.5, label='100%')
ax1.set_xlabel('Diferença relativa entre C0 e C1 (%)', fontsize=11)
ax1.set_ylabel('Accuracy total (%)', fontsize=11)
ax1.set_title('Sensibilidade: Accuracy vs Perturbação', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0, 105])

# Gráfico 2: Acerto individual dos consumidores 0 e 1
ax2.plot(eps_vals * 100, c0_correct.astype(float) * 100, 'o-', linewidth=2, markersize=8, label='Consumidor 0 (Fase 1)')
ax2.plot(eps_vals * 100, c1_correct.astype(float) * 100, 's-', linewidth=2, markersize=8, label='Consumidor 1 (Fase 2)')
ax2.set_xlabel('Diferença relativa entre C0 e C1 (%)', fontsize=11)
ax2.set_ylabel('Acerto (%)', fontsize=11)
ax2.set_title('Sensibilidade: Acerto de C0 e C1', fontsize=12)
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, 105])

plt.tight_layout()
plt.savefig('/home/matilde/Desktop/ADRI/1/sensitivity_analysis.png', dpi=100)
print("\nGráfico guardado em: sensitivity_analysis.png")
plt.show()

# ============================================================
# INTERPRETAÇÃO
# ============================================================

print("\n" + "="*70)
print("INTERPRETAÇÃO")
print("="*70)

if c0_correct[0] and c1_correct[0]:
    print(f"\nCom epsilon=0: Os dois consumidores conseguem ser bem distinguidos.")
else:
    print(f"\nCom epsilon=0: Problema detectado (esperado).")

threshold_idx = np.where(~c0_correct | ~c1_correct)[0]
if len(threshold_idx) > 0:
    critical_eps = eps_vals[threshold_idx[0]]
    print(f"\nLimiar crítico: ε ≈ {critical_eps*100:.3f}%")
    print(f"  → A partir daqui, começa a haver erros na atribuição de fases")
else:
    print(f"\nO método consegue distinguir os consumidores mesmo com perturbações até {eps_vals[-1]*100:.2f}%")
