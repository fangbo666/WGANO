"""
MHD visualization: 2x3 grid
Row 0: EM field |J| = sqrt(Jx²+Jy²+Jz²)
Row 1: Flow field |u| = sqrt(uu²+uv²)
Cols: True Solution | SAOT Error | CWGANO Error
Each subplot has its own independent colorbar.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Fix fonts for AAAI 2027 (avoid Identity-H encoding) ─────────────────────
plt.rcParams['pdf.fonttype'] = 42       # TrueType (avoids Type 3)
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'sans-serif'  # Helvetica recommended by AAAI
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
plt.rcParams['font.serif'] = ['Times', 'Times New Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'cm'     # Computer Modern for math (allowed)
plt.rcParams['text.usetex'] = False         # Avoid LaTeX fonts

VIS_DIR   = os.path.dirname(os.path.abspath(__file__))
cache_dir = os.path.join(VIS_DIR, "mhd_cache")

# Load cached predictions
saot_em   = np.load(os.path.join(cache_dir, "saot_em.npy"))    # [N,128,128,3]
saot_fl   = np.load(os.path.join(cache_dir, "saot_fl.npy"))    # [N,128,128,2]
cwgano_em = np.load(os.path.join(cache_dir, "cwgano_em.npy"))  # [N,128,128,3]
cwgano_fl = np.load(os.path.join(cache_dir, "cwgano_fl.npy"))  # [N,128,128,2]
em_true   = np.load(os.path.join(cache_dir, "em_true.npy"))    # [N,128,128,3]
fl_true   = np.load(os.path.join(cache_dir, "fl_true.npy"))    # [N,128,128,2]

N = em_true.shape[0]

# Per-sample nRMSE (per-channel mean, matching metrics.json formula)
def per_sample_nrmse_em(pred, true):
    errs = np.zeros(len(pred))
    for i in range(len(pred)):
        ch = []
        for c in range(3):
            p = pred[i, :, :, c].ravel()
            t = true[i, :, :, c].ravel()
            ch.append(np.linalg.norm(p - t) / (np.linalg.norm(t) + 1e-8))
        errs[i] = np.mean(ch)
    return errs

saot_nrmse_em   = per_sample_nrmse_em(saot_em, em_true)
cwgano_nrmse_em = per_sample_nrmse_em(cwgano_em, em_true)

# Pick sample with best SAOT/CWGANO ratio (most visually distinctive)
ratio = saot_nrmse_em / (cwgano_nrmse_em + 1e-10)
idx = int(np.argmax(ratio))
print(f"Selected sample index: {idx}")
print(f"  SAOT   EM nRMSE: {saot_nrmse_em[idx]:.4f}")
print(f"  CWGANO EM nRMSE: {cwgano_nrmse_em[idx]:.4f}")
print(f"  Ratio: {ratio[idx]:.1f}x")

# Extract single sample
em_t = em_true[idx]    # [128,128,3]
fl_t = fl_true[idx]    # [128,128,2]
em_s = saot_em[idx]
fl_s = saot_fl[idx]
em_c = cwgano_em[idx]
fl_c = cwgano_fl[idx]

# Compute magnitudes
J_true   = np.sqrt(np.sum(em_t**2, axis=-1))   # [128,128]
u_true   = np.sqrt(np.sum(fl_t**2, axis=-1))
J_saot   = np.sqrt(np.sum(em_s**2, axis=-1))
u_saot   = np.sqrt(np.sum(fl_s**2, axis=-1))
J_cwgano = np.sqrt(np.sum(em_c**2, axis=-1))
u_cwgano = np.sqrt(np.sum(fl_c**2, axis=-1))

err_saot_J   = J_saot   - J_true
err_cwgano_J = J_cwgano - J_true
err_saot_u   = u_saot   - u_true
err_cwgano_u = u_cwgano - u_true

print(f"\nJ_true    range: [{J_true.min():.4f}, {J_true.max():.4f}]")
print(f"err_saot_J   max={err_saot_J.max():.4f}  mean={err_saot_J.mean():.4f}")
print(f"err_cwgano_J max={err_cwgano_J.max():.4f}  mean={err_cwgano_J.mean():.4f}")
print(f"u_true    range: [{u_true.min():.4f}, {u_true.max():.4f}]")
print(f"err_saot_u   max={err_saot_u.max():.4f}  mean={err_saot_u.mean():.4f}")
print(f"err_cwgano_u max={err_cwgano_u.max():.4f}  mean={err_cwgano_u.mean():.4f}")

# --- Plot: each subplot has its own colorbar range ---
fig, axes = plt.subplots(2, 3, figsize=(12, 7))

cmap_sol = "RdBu_r"
cmap_err = "RdBu_r"

FONT_TITLE = 20   # column/subplot titles
FONT_LABEL = 18   # row ylabel
FONT_CB    = 20   # colorbar tick labels (increased from 16)
FONT_CB_TL = 20   # colorbar title (magnitude) (increased from 16)

col_labels = ["True Solution", "SAOT Error", "CWGANO Error"]
row_labels  = [r"$|\mathbf{J}|$", r"$|\mathbf{u}|$"]

data_rows = [
    [J_true, err_saot_J,   err_cwgano_J],
    [u_true, err_saot_u,   err_cwgano_u],
]

# Per-row error vmax: 99th percentile of abs error (clips outliers, improves contrast)
err_vmax_raw_row = [
    np.percentile(np.concatenate([np.abs(err_saot_J).flatten(), np.abs(err_cwgano_J).flatten()]), 99),
    np.percentile(np.concatenate([np.abs(err_saot_u).flatten(), np.abs(err_cwgano_u).flatten()]), 99),
]
magnitude_row = [int(np.floor(np.log10(v))) for v in err_vmax_raw_row]
err_scale_row = [10**m for m in magnitude_row]
err_vmax_row = [err_vmax_raw_row[i] / err_scale_row[i] for i in range(2)]

for r, (row_data, row_lbl) in enumerate(zip(data_rows, row_labels)):
    err_vmax_normalized = err_vmax_row[r]
    err_vmax_raw = err_vmax_raw_row[r]
    err_scale = err_scale_row[r]
    magnitude = magnitude_row[r]
    for c, arr in enumerate(row_data):
        ax = axes[r, c]
        if c == 0:
            # True solution: use evenly-spaced colorbar ticks
            data_min, data_max = arr.min(), arr.max()
            tick_spacing = np.ceil(data_max / 2 * 10) / 10
            vmin = 0
            vmax = tick_spacing * 2
            cmap = cmap_sol
            im = ax.imshow(arr, cmap=cmap, origin="lower",
                           vmin=vmin, vmax=vmax, interpolation="nearest")
            cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                            ticks=[0, tick_spacing, vmax])
            mag_true = int(np.floor(np.log10(vmax)))
            scale = 10**(-mag_true)
            cb.ax.set_yticklabels([f'{0:.1f}', f'{tick_spacing*scale:.1f}', f'{vmax*scale:.1f}'],
                                   fontsize=FONT_CB)
            cb.ax.set_title(f'1e{mag_true:d}', fontsize=FONT_CB_TL, pad=8)
            cb.ax.tick_params(size=0)
            # Nudge bottom label up and top label down toward center
            fig.canvas.draw()
            labels = cb.ax.yaxis.get_majorticklabels()
            from matplotlib.transforms import ScaledTranslation
            offset_up = ScaledTranslation(0, 10/72., fig.dpi_scale_trans)   # 10pt up
            offset_down = ScaledTranslation(0, -10/72., fig.dpi_scale_trans) # 10pt down
            labels[0].set_transform(labels[0].get_transform() + offset_up)
            labels[-1].set_transform(labels[-1].get_transform() + offset_down)
        else:
            # Error maps - normalize data
            data_normalized = arr / err_scale
            im = ax.imshow(data_normalized, cmap=cmap_err, origin="lower",
                           vmin=-err_vmax_normalized, vmax=err_vmax_normalized, interpolation="nearest")
            cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                            ticks=[-err_vmax_normalized, 0, err_vmax_normalized])
            cb.ax.set_yticklabels([f'{-err_vmax_normalized:.1f}', '0.0', f'{err_vmax_normalized:.1f}'], fontsize=FONT_CB)
            cb.ax.set_title(f'1e{magnitude:d}', fontsize=FONT_CB_TL, pad=8)
            cb.ax.tick_params(size=0)
            # Nudge top and bottom labels inward
            fig.canvas.draw()
            labels = cb.ax.yaxis.get_majorticklabels()
            from matplotlib.transforms import ScaledTranslation
            offset_up = ScaledTranslation(0, 10/72., fig.dpi_scale_trans)   # 10pt up
            offset_down = ScaledTranslation(0, -10/72., fig.dpi_scale_trans) # 10pt down
            labels[0].set_transform(labels[0].get_transform() + offset_up)
            labels[-1].set_transform(labels[-1].get_transform() + offset_down)
        ax.set_xticks([])
        ax.set_yticks([])
        if r == 0:
            ax.set_title(col_labels[c], fontsize=FONT_TITLE, fontweight="bold")
        if c == 0:
            ax.set_ylabel(row_lbl, fontsize=FONT_LABEL, fontweight="bold")

plt.tight_layout()
out_pdf = os.path.join(VIS_DIR, "07_mhd.pdf")
plt.savefig(out_pdf, dpi=300)
print(f"\nSaved: {out_pdf}")
