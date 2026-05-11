# =============================================================================
# MANTENIMIENTO PREDICTIVO - COMPARATIVA DE ALGORITMOS ML
# Dataset: NASA C-MAPSS (train_FD001.txt)
# Muestra 5 graficas en pantalla al ejecutar (sin generar Word)
# =============================================================================
# pip install numpy pandas matplotlib seaborn scikit-learn
# =============================================================================

import os, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report
)

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size':   10,
    'axes.titlesize':   11,
    'axes.titleweight': 'bold',
    'figure.dpi': 120
})

COL = {
    'Decision Tree':    '#2196F3',
    'SVM (RBF)':        '#F44336',
    'Red Neuronal MLP': '#4CAF50'
}

# =============================================================================
# 1. CARGA DE DATOS
# =============================================================================
print("=" * 65)
print("  MANTENIMIENTO PREDICTIVO - NASA C-MAPSS FD001")
print("=" * 65)
print("\n[1/5] Cargando datos...")

ARCHIVO = 'data/train_FD001.txt'
if not os.path.exists(ARCHIVO):
    ARCHIVO = 'train_FD001.txt'  # fallback

columnas = (['engine_id', 'cycle'] +
            [f'setting_{i}' for i in range(1, 4)] +
            [f'sensor_{i}'  for i in range(1, 22)])
df = pd.read_csv(ARCHIVO, sep=r'\s+', header=None,
                 names=columnas, index_col=False)
df.dropna(axis=1, how='all', inplace=True)

max_c = df.groupby('engine_id')['cycle'].max().rename('max_cycle')
df    = df.join(max_c, on='engine_id')
df['RUL']   = df['max_cycle'] - df['cycle']
df['label'] = (df['RUL'] <= 30).astype(int)
cnt = df['label'].value_counts()

print(f"  {df.shape[0]:,} filas x {df.shape[1]} col | "
      f"{df['engine_id'].nunique()} motores | "
      f"Ratio desbalanceo {cnt[0]/cnt[1]:.1f}:1")

# =============================================================================
# 2. FEATURES, SPLIT Y ESCALADO
# =============================================================================
print("[2/5] Preprocesando...")
COLS_ELIM = ['engine_id','cycle','max_cycle','RUL',
             'setting_1','setting_2','setting_3',
             'sensor_1','sensor_5','sensor_10',
             'sensor_16','sensor_18','sensor_19']
COLS_ELIM = [c for c in COLS_ELIM if c in df.columns]
X = df.drop(columns=COLS_ELIM + ['label'])
y = df['label']
nombres_feat = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)
sc  = StandardScaler()
Xtr = sc.fit_transform(X_train)
Xte = sc.transform(X_test)

# =============================================================================
# 3. ENTRENAMIENTO
# =============================================================================
print("[3/5] Entrenando modelos...")
m_dt  = DecisionTreeClassifier(max_depth=8, class_weight='balanced', random_state=42)
m_svm = SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42)
m_mlp = MLPClassifier(hidden_layer_sizes=(64,32), activation='relu', solver='adam',
                      max_iter=500, random_state=42,
                      early_stopping=True, n_iter_no_change=15)
m_dt.fit(Xtr, y_train)
m_svm.fit(Xtr, y_train)
m_mlp.fit(Xtr, y_train)
modelos = {'Decision Tree': m_dt, 'SVM (RBF)': m_svm, 'Red Neuronal MLP': m_mlp}

# =============================================================================
# 4. EVALUACION
# =============================================================================
print("[4/5] Evaluando...")
res = {}
for n, m in modelos.items():
    yp  = m.predict(Xte)
    ypr = m.predict_proba(Xte)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, ypr)
    cr  = classification_report(y_test, yp,
          target_names=['Sano (0)', 'Fallo inminente (1)'], output_dict=True)
    res[n] = dict(
        y_pred=yp, y_prob=ypr, cm=confusion_matrix(y_test, yp),
        fpr=fpr, tpr=tpr, auc=float(auc(fpr, tpr)),
        accuracy=float(accuracy_score(y_test, yp)),
        precision=float(precision_score(y_test, yp, zero_division=0)),
        recall=float(recall_score(y_test, yp)),
        f1=float(f1_score(y_test, yp)), cr=cr,
    )
    print(f"  {n:<22} Acc={res[n]['accuracy']:.3f}  "
          f"AUC={res[n]['auc']:.3f}  Rec={res[n]['recall']:.3f}  "
          f"F1={res[n]['f1']:.3f}")

mejor    = max(res, key=lambda k: res[k]['auc'])
nombres  = list(res.keys())
best_idx = nombres.index(mejor)
unif     = 1 / len(nombres_feat)

imp    = m_dt.feature_importances_
df_imp = pd.DataFrame({'Sensor': nombres_feat, 'Imp': imp}).sort_values('Imp')
top5   = df_imp.sort_values('Imp', ascending=False).head(5)

print(f"\n  Mejor modelo: {mejor} (AUC={res[mejor]['auc']:.4f})")
print("[5/5] Generando graficas...\n")

# =============================================================================
# GRAFICA 1 — MATRICES DE CONFUSION
# =============================================================================
CMAP_NAVY = LinearSegmentedColormap.from_list(
    'navy_custom', ['#E3F2FD', '#0D2B4E'], N=256)

fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.patch.set_facecolor('#F5F5F5')
fig.suptitle('Figura 1 — Matrices de Confusion sobre el Conjunto de Prueba',
             fontsize=13, fontweight='bold', y=1.01, color='#0D2B4E')

etq  = ['Sano (0)', 'Fallo (1)']
TIPO = {(0,0):'VN', (0,1):'FP', (1,0):'FN', (1,1):'VP'}

for ax, n in zip(axes, nombres):
    cm_v  = res[n]['cm']
    cm_p  = cm_v.astype(float) / cm_v.sum(axis=1, keepdims=True) * 100
    total = cm_v.sum()

    sns.heatmap(cm_v, annot=False, cmap=CMAP_NAVY, ax=ax,
                linewidths=2.5, linecolor='#F5F5F5', cbar=False,
                xticklabels=etq, yticklabels=etq)

    for i in range(2):
        for j in range(2):
            val    = cm_v[i, j]
            bright = val < total * 0.25
            fc     = '#0D2B4E' if bright else 'white'
            ax.text(j+0.5, i+0.28, TIPO[(i,j)],
                    ha='center', va='center', fontsize=10,
                    color=fc, alpha=0.55, fontweight='bold')
            ax.text(j+0.5, i+0.52, str(val),
                    ha='center', va='center', fontsize=22,
                    color=fc, fontweight='bold')
            ax.text(j+0.5, i+0.76, f'{cm_p[i,j]:.1f}%',
                    ha='center', va='center', fontsize=10,
                    color=fc, alpha=0.85)

    ax.set_title(n, fontsize=11, fontweight='bold', color=COL[n], pad=12)
    ax.set_xlabel('Clase Predicha', fontsize=9, labelpad=6, color='#455A64')
    ax.set_ylabel('Clase Real',     fontsize=9, labelpad=6, color='#455A64')
    ax.tick_params(labelsize=9, length=0)

    TP = cm_v[1,1]; TN = cm_v[0,0]
    FP = cm_v[0,1]; FN = cm_v[1,0]
    prec = TP/(TP+FP) if (TP+FP) > 0 else 0
    rec  = TP/(TP+FN) if (TP+FN) > 0 else 0
    f1_v = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0
    txt  = (f"Acc={res[n]['accuracy']:.3f}  AUC={res[n]['auc']:.3f}  "
            f"Prec={prec:.3f}  Rec={rec:.3f}  F1={f1_v:.3f}")
    ax.text(1.0, -0.18, txt, transform=ax.transAxes,
            ha='center', fontsize=7.8, color='#546E7A', style='italic',
            bbox=dict(facecolor='white', alpha=0.7,
                      edgecolor='#B0BEC5', boxstyle='round,pad=0.3'))
    for spine in ax.spines.values():
        spine.set_visible(False)

plt.tight_layout(pad=1.8)
plt.show()

# =============================================================================
# GRAFICA 2 — REPORTE DE CLASIFICACION (heatmap por clase)
# =============================================================================
clases_rep = ['Sano (0)', 'Fallo inminente (1)', 'macro avg', 'weighted avg']
met_rep    = ['precision', 'recall', 'f1-score']
met_labels = ['Precision', 'Recall', 'F1-Score']

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor('white')
fig.suptitle('Figura 2 — Reporte de Clasificacion por Modelo y Clase',
             fontsize=13, fontweight='bold', y=1.01, color='#0D2B4E')

for ax, n in zip(axes, nombres):
    cr     = res[n]['cr']
    matrix = np.array([[cr.get(cls, {}).get(m, 0)
                        for m in met_rep]
                       for cls in clases_rep])

    im = ax.imshow(matrix, cmap='YlGn', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(3)); ax.set_xticklabels(met_labels, fontsize=9)
    ax.set_yticks(range(4)); ax.set_yticklabels(clases_rep, fontsize=9)

    for i in range(4):
        for j in range(3):
            val = matrix[i, j]
            fc  = 'white' if val > 0.6 else '#1A237E'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                    fontsize=11, fontweight='bold', color=fc)

    ax.set_title(n, fontsize=11, fontweight='bold', color=COL[n], pad=10)
    plt.colorbar(im, ax=ax, shrink=0.8)

plt.tight_layout(pad=1.5)
plt.show()

# =============================================================================
# GRAFICA 3 — CURVAS ROC COMPARATIVAS
# =============================================================================
fig, ax = plt.subplots(figsize=(8, 7))
fig.patch.set_facecolor('white')

ax.plot([0,1],[0,1], '--', color='#9E9E9E', lw=1.4,
        label='Clasificador aleatorio (AUC=0.50)', alpha=0.7, zorder=1)

for n, r in res.items():
    fpr_a, tpr_a = np.array(r['fpr']), np.array(r['tpr'])
    ax.plot(fpr_a, tpr_a, color=COL[n], lw=2.5,
            label=f"{n}  (AUC={r['auc']:.4f})", zorder=3)
    j = np.argmax(tpr_a - fpr_a)
    ax.scatter(fpr_a[j], tpr_a[j], color=COL[n], s=90, zorder=5,
               edgecolors='white', linewidths=1.5)

ax.fill_between(res[mejor]['fpr'], res[mejor]['tpr'],
                alpha=0.08, color=COL[mejor])

ax.set_title('Figura 3 — Curvas ROC Comparativas',
             fontsize=13, fontweight='bold', pad=14, color='#0D2B4E')
ax.set_xlabel('Tasa de Falsos Positivos (FPR)', fontsize=11)
ax.set_ylabel('Tasa de Verdaderos Positivos (TPR)', fontsize=11)
ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
ax.legend(loc='lower right', fontsize=9.5, framealpha=0.93)
ax.grid(True, ls='--', alpha=0.35)
ax.annotate('Zona ideal', xy=(0.04, 0.94), fontsize=9,
            color='#2E7D32', style='italic')
ax.annotate('Azar', xy=(0.55, 0.44), fontsize=9,
            color='#757575', style='italic', rotation=34)
for spine in ['top','right']:
    ax.spines[spine].set_visible(False)
plt.tight_layout()
plt.show()

# =============================================================================
# GRAFICA 4 — IMPORTANCIA DE CARACTERISTICAS
# =============================================================================
cols_b = plt.cm.Blues(np.linspace(0.28, 0.95, len(df_imp)))
fig, ax = plt.subplots(figsize=(9, max(5, len(df_imp) * 0.50)))
fig.patch.set_facecolor('white')

bars = ax.barh(df_imp['Sensor'], df_imp['Imp'],
               color=cols_b, edgecolor='white', height=0.72)
for b, v in zip(bars, df_imp['Imp']):
    if v > 0.003:
        ax.text(v + 0.003, b.get_y() + b.get_height()/2,
                f'{v:.3f}', va='center', ha='left',
                fontsize=8.5, color='#263238')

ax.axvline(unif, color='#E53935', ls='--', lw=1.4, alpha=0.85,
           label=f'Importancia uniforme (1/{len(nombres_feat)} = {unif:.3f})')
ax.set_xlabel('Importancia (reduccion impureza Gini normalizada)', fontsize=10)
ax.set_xlim([0, df_imp['Imp'].max() * 1.22])
ax.set_title('Figura 4 — Importancia de Caracteristicas (Decision Tree)',
             fontsize=12, fontweight='bold', pad=12, color='#0D2B4E')
ax.legend(fontsize=9, loc='lower right')
ax.grid(True, axis='x', ls='--', alpha=0.35)
for spine in ['top','right']:
    ax.spines[spine].set_visible(False)
plt.tight_layout()
plt.show()

# =============================================================================
# GRAFICA 5 — DASHBOARD COMPARATIVO DE METRICAS
# =============================================================================
met_cols = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
df_met   = pd.DataFrame([{
    'Modelo': n, 'Accuracy': r['accuracy'], 'Precision': r['precision'],
    'Recall': r['recall'], 'F1-Score': r['f1'], 'AUC-ROC': r['auc']
} for n, r in res.items()])

x, ancho, offsets = np.arange(len(met_cols)), 0.23, [-0.23, 0, 0.23]

fig, ax = plt.subplots(figsize=(13, 6))
fig.patch.set_facecolor('white')

for n, off in zip(df_met['Modelo'], offsets):
    vals  = df_met.loc[df_met['Modelo']==n, met_cols].values[0]
    barrs = ax.bar(x+off, vals, ancho, label=n,
                   color=COL[n], alpha=0.88, edgecolor='white', lw=0.7)
    for b, v in zip(barrs, vals):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.009,
                f'{v:.3f}', ha='center', va='bottom',
                fontsize=8, fontweight='bold', color=COL[n])

best_off = offsets[best_idx]
ax.axvspan(x[0]+best_off - ancho/2 - 0.05,
           x[-1]+best_off + ancho/2 + 0.05,
           alpha=0.07, color=COL[mejor])

ax.set_title('Figura 5 — Dashboard Comparativo de Metricas',
             fontsize=13, fontweight='bold', pad=14, color='#0D2B4E')
ax.set_xlabel('Metrica de Evaluacion', fontsize=11)
ax.set_ylabel('Valor (0 - 1)', fontsize=11)
ax.set_xticks(x); ax.set_xticklabels(met_cols, fontsize=10.5)
ax.set_ylim([0, 1.15])
ax.axhline(1.0, color='#9E9E9E', ls='--', lw=0.9, alpha=0.5)
ax.legend(loc='lower right', fontsize=10, framealpha=0.93)
ax.grid(True, axis='y', ls='--', alpha=0.35)
for spine in ['top','right']:
    ax.spines[spine].set_visible(False)
plt.tight_layout()
plt.show()

# =============================================================================
# RESUMEN EN CONSOLA
# =============================================================================
print()
print("=" * 65)
print("  RESUMEN FINAL")
print("=" * 65)
print(f"  {'Modelo':<22} {'Acc':>7} {'Prec':>7} {'Rec':>7} "
      f"{'F1':>7} {'AUC':>7}")
print(f"  {'-'*57}")
for n in nombres:
    r     = res[n]
    marca = '  <-- MEJOR' if n == mejor else ''
    print(f"  {n:<22} {r['accuracy']:>7.4f} {r['precision']:>7.4f} "
          f"{r['recall']:>7.4f} {r['f1']:>7.4f} {r['auc']:>7.4f}{marca}")
print()
print(f"  Top 3 sensores mas predictivos (Decision Tree):")
for i, (_, row) in enumerate(top5.head(3).iterrows(), 1):
    print(f"    {i}. {row['Sensor']:<12}  imp = {row['Imp']:.4f}")
print()
print("  Graficas mostradas:")
print("    [1] Matrices de confusion con VP/VN/FP/FN y metricas")
print("    [2] Reporte de clasificacion por clase (heatmap)")
print("    [3] Curvas ROC comparativas con umbral de Youden")
print("    [4] Importancia de caracteristicas (Decision Tree)")
print("    [5] Dashboard comparativo de metricas")
print("=" * 65)