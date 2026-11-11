"""
通用可视化框架 - 支持单物理场(1x3)和多物理场(2x3)布局
遵循AAAI 2027字体规范
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.transforms import ScaledTranslation

# ── AAAI 2027字体配置 (避免Identity-H encoding) ───────────────────────────
plt.rcParams['pdf.fonttype'] = 42       # TrueType (avoids Type 3)
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.serif'] = ['Times', 'Times New Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['text.usetex'] = False

# ── 字体大小常量 ──────────────────────────────────────────────────────────
FONT_TITLE = 20   # 列标题
FONT_LABEL = 18   # 行标签
FONT_CB    = 20   # colorbar刻度
FONT_CB_TL = 20   # colorbar magnitude标题


def visualize_single_physics(
    true_sol,
    saot_pred,
    wgano_pred,
    field_name="Field",
    output_path="output.pdf",
    png_path=None,
    cmap_sol="RdBu_r",
    cmap_err="RdBu_r"
):
    """
    单物理场可视化: 1行3列布局

    Args:
        true_sol: [H, W] 真实解
        saot_pred: [H, W] SAOT预测
        wgano_pred: [H, W] WGANO预测
        field_name: 物理场名称 (如 "p", "|u|", "σ")
        output_path: PDF输出路径
        png_path: PNG输出路径 (可选)
        cmap_sol: true solution的colormap
        cmap_err: error的colormap
    """
    err_saot = saot_pred - true_sol
    err_wgano = wgano_pred - true_sol

    # 计算error的统一vmax (99th percentile避免outlier)
    err_vmax = np.percentile(np.maximum(np.abs(err_saot), np.abs(err_wgano)), 99)
    err_magnitude = int(np.floor(np.log10(err_vmax)))

    # 创建figure
    fig = plt.figure(figsize=(15, 4))
    gs = GridSpec(1, 3, figure=fig, wspace=0.35)

    col_labels = ["True Solution", "SAOT Error", "WGANO Error"]
    data_list = [true_sol, err_saot, err_wgano]

    for c, (data, label) in enumerate(zip(data_list, col_labels)):
        ax = fig.add_subplot(gs[c])

        if c == 0:
            # True solution
            vmin, vmax = data.min(), data.max()
            tick_spacing = np.ceil(vmax / 2 * 10) / 10
            cb_ticks = [0, tick_spacing, tick_spacing * 2]

            im = ax.imshow(data, cmap=cmap_sol, origin='lower',
                          vmin=0, vmax=cb_ticks[-1], interpolation='nearest')
            ax.set_title(label, fontsize=20, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
            cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=cb_ticks)

            # Magnitude和刻度
            mag_true = int(np.floor(np.log10(cb_ticks[-1])))
            scale = 10**(-mag_true)
            cb.ax.set_yticklabels([f'{t*scale:.1f}' for t in cb_ticks], fontsize=FONT_CB)
            cb.ax.set_title(f'1e{mag_true:d}', fontsize=FONT_CB_TL, pad=8)
            cb.ax.tick_params(size=0)

        else:
            # Error maps
            im = ax.imshow(data, cmap=cmap_err, origin='lower',
                          vmin=-err_vmax, vmax=err_vmax, interpolation='nearest')
            ax.set_title(label, fontsize=20, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
            cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                            ticks=[-err_vmax, 0, err_vmax])
            cb.ax.set_yticklabels([f'{-err_vmax:.1f}', '0.0', f'{err_vmax:.1f}'], fontsize=FONT_CB)
            cb.ax.set_title(f'1e{err_magnitude:d}', fontsize=FONT_CB_TL, pad=8)
            cb.ax.tick_params(size=0)

        # 微调colorbar刻度标签位置(向中心靠拢)
        fig.canvas.draw()
        labels = cb.ax.yaxis.get_majorticklabels()
        offset_up = ScaledTranslation(0, 10/72., fig.dpi_scale_trans)
        offset_down = ScaledTranslation(0, -10/72., fig.dpi_scale_trans)
        labels[0].set_transform(labels[0].get_transform() + offset_up)
        labels[-1].set_transform(labels[-1].get_transform() + offset_down)

        # if c == 0:
        #     ax.set_ylabel(field_name, fontsize=FONT_LABEL, fontweight="bold")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved PDF: {output_path}")

    if png_path:
        fig.savefig(png_path, dpi=150, bbox_inches='tight')
        print(f"Saved PNG: {png_path}")

    plt.close(fig)


def visualize_multiphysics(
    field_a_true,
    field_a_saot,
    field_a_wgano,
    field_b_true,
    field_b_saot,
    field_b_wgano,
    field_a_name=r"Field A",
    field_b_name=r"Field B",
    output_path="output.pdf",
    png_path=None,
    cmap_sol="RdBu_r",
    cmap_err="RdBu_r"
):
    """
    多物理场可视化: 2行3列布局

    Args:
        field_a_true: [H, W] 物理场A真实解
        field_a_saot: [H, W] 物理场A SAOT预测
        field_a_wgano: [H, W] 物理场A WGANO预测
        field_b_true: [H, W] 物理场B真实解
        field_b_saot: [H, W] 物理场B SAOT预测
        field_b_wgano: [H, W] 物理场B WGANO预测
        field_a_name: 物理场A名称 (如 r"$|\mathbf{J}|$")
        field_b_name: 物理场B名称 (如 r"$|\mathbf{u}|$")
        output_path: PDF输出路径
        png_path: PNG输出路径
        cmap_sol: colormap for solution
        cmap_err: colormap for error
    """
    # 计算误差
    err_saot_a = field_a_saot - field_a_true
    err_wgano_a = field_a_wgano - field_a_true
    err_saot_b = field_b_saot - field_b_true
    err_wgano_b = field_b_wgano - field_b_true

    # 每行的error vmax (99th percentile)
    err_vmax_a = np.percentile(np.maximum(np.abs(err_saot_a), np.abs(err_wgano_a)), 99)
    err_vmax_b = np.percentile(np.maximum(np.abs(err_saot_b), np.abs(err_wgano_b)), 99)
    err_mag_a = int(np.floor(np.log10(err_vmax_a)))
    err_mag_b = int(np.floor(np.log10(err_vmax_b)))

    # 创建figure: 2行3列
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))

    col_labels = ["True Solution", "SAOT Error", "WGANO Error"]
    row_labels = [field_a_name, field_b_name]

    data_rows = [
        [field_a_true, err_saot_a, err_wgano_a],
        [field_b_true, err_saot_b, err_wgano_b],
    ]
    err_vmaxs = [err_vmax_a, err_vmax_b]
    err_mags = [err_mag_a, err_mag_b]

    for r, (row_data, row_label, err_vmax, err_mag) in enumerate(
        zip(data_rows, row_labels, err_vmaxs, err_mags)
    ):
        for c, data in enumerate(row_data):
            ax = axes[r, c]

            if c == 0:
                # True solution
                vmin, vmax = data.min(), data.max()
                tick_spacing = np.ceil(vmax / 2 * 10) / 10
                cb_vmax = tick_spacing * 2

                im = ax.imshow(data, cmap=cmap_sol, origin='lower',
                              vmin=0, vmax=cb_vmax, interpolation='nearest')
                ax.set_title(col_labels[c], fontsize=FONT_TITLE, fontweight="bold")
                ax.set_xticks([])
                ax.set_yticks([])
                if c == 0:
                    ax.set_ylabel(row_label, fontsize=18, fontweight='bold')
                cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                                ticks=[0, tick_spacing, cb_vmax])

                mag_true = int(np.floor(np.log10(cb_vmax)))
                scale = 10**(-mag_true)
                cb.ax.set_yticklabels([f'{0:.1f}', f'{tick_spacing*scale:.1f}',
                                       f'{cb_vmax*scale:.1f}'], fontsize=FONT_CB)
                cb.ax.set_title(f'1e{mag_true:d}', fontsize=FONT_CB_TL, pad=8)
                cb.ax.tick_params(size=0)

            else:
                # Error maps
                im = ax.imshow(data, cmap=cmap_err, origin='lower',
                              vmin=-err_vmax, vmax=err_vmax, interpolation='nearest')
                ax.set_title(col_labels[c], fontsize=FONT_TITLE, fontweight="bold")
                ax.set_xticks([])
                ax.set_yticks([])
                if c == 0:
                    ax.set_ylabel(row_label, fontsize=18, fontweight='bold')
                cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                                ticks=[-err_vmax, 0, err_vmax])
                cb.ax.set_yticklabels([f'{-err_vmax:.1f}', '0.0', f'{err_vmax:.1f}'], fontsize=FONT_CB)
                cb.ax.set_title(f'1e{err_mag:d}', fontsize=FONT_CB_TL, pad=8)
                cb.ax.tick_params(size=0)

            # 微调colorbar刻度标签
            fig.canvas.draw()
            labels = cb.ax.yaxis.get_majorticklabels()
            offset_up = ScaledTranslation(0, 10/72., fig.dpi_scale_trans)
            offset_down = ScaledTranslation(0, -10/72., fig.dpi_scale_trans)
            labels[0].set_transform(labels[0].get_transform() + offset_up)
            labels[-1].set_transform(labels[-1].get_transform() + offset_down)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved PDF: {output_path}")

    if png_path:
        fig.savefig(png_path, dpi=150, bbox_inches='tight')
        print(f"Saved PNG: {png_path}")

    plt.close(fig)


def compute_per_sample_nrmse(pred, true, n_channels=1):
    """
    计算per-sample nRMSE (多通道平均)

    Args:
        pred: [N, H, W, C] 或 [N, H, W] 预测
        true: [N, H, W, C] 或 [N, H, W] 真实
        n_channels: 通道数

    Returns:
        errs: [N] 每个样本的nRMSE
    """
    if pred.ndim == 3:
        pred = pred[..., None]
        true = true[..., None]

    N = len(pred)
    errs = np.zeros(N)

    for i in range(N):
        ch_errs = []
        for c in range(n_channels):
            p = pred[i, :, :, c].ravel()
            t = true[i, :, :, c].ravel()
            ch_errs.append(np.linalg.norm(p - t) / (np.linalg.norm(t) + 1e-8))
        errs[i] = np.mean(ch_errs)

    return errs


def select_best_sample(saot_pred, wgano_pred, true_sol, n_channels=1):
    """
    选择SAOT/WGANO ratio最大的样本 (最能展示差异)

    Returns:
        idx: 最佳样本索引
        ratio: SAOT/WGANO nRMSE比值
    """
    saot_nrmse = compute_per_sample_nrmse(saot_pred, true_sol, n_channels)
    wgano_nrmse = compute_per_sample_nrmse(wgano_pred, true_sol, n_channels)

    ratio = saot_nrmse / (wgano_nrmse + 1e-10)
    idx = int(np.argmax(ratio))

    print(f"Selected sample {idx}: SAOT nRMSE={saot_nrmse[idx]:.4f}, "
          f"WGANO nRMSE={wgano_nrmse[idx]:.4f}, Ratio={ratio[idx]:.2f}x")

    return idx, ratio[idx]


def visualize_coupled_physics(
    true_flow, true_T,
    saot_flow, saot_T,
    cwgano_flow, cwgano_T,
    output_path="output.pdf",
    png_path=None
):
    """
    多物理场耦合可视化: 2行3列布局
    第1行: Flow场 (|u|)
    第2行: 温度场 T

    Args:
        true_flow: [H, W, 2] 真实流场 (u, v)
        true_T: [H, W, 1] 真实温度
        saot_flow: [H, W, 2] SAOT流场预测
        saot_T: [H, W, 1] SAOT温度预测
        cwgano_flow: [H, W, 2] CWGANO流场预测
        cwgano_T: [H, W, 1] CWGANO温度预测
    """
    # 计算流场magnitude
    true_flow_mag = np.sqrt(true_flow[..., 0]**2 + true_flow[..., 1]**2)
    saot_flow_mag = np.sqrt(saot_flow[..., 0]**2 + saot_flow[..., 1]**2)
    cwgano_flow_mag = np.sqrt(cwgano_flow[..., 0]**2 + cwgano_flow[..., 1]**2)

    # 提取温度
    true_T_2d = true_T.squeeze()
    saot_T_2d = saot_T.squeeze()
    cwgano_T_2d = cwgano_T.squeeze()

    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.35,
                  left=0.08, right=0.95, top=0.92, bottom=0.08)

    # 第1行: Flow
    for col, (data, title) in enumerate([
        (true_flow_mag, 'Ground Truth'),
        (saot_flow_mag - true_flow_mag, 'SAOT Error'),
        (cwgano_flow_mag - true_flow_mag, 'CWGANO Error')
    ]):
        ax = fig.add_subplot(gs[0, col])

        if col == 0:
            im = ax.imshow(data, cmap='RdBu_r', origin='lower')
            vmin, vmax = data.min(), data.max()
        else:
            vmax_err = max(abs(data.min()), abs(data.max()))
            im = ax.imshow(data, cmap='RdBu_r', origin='lower', vmin=-vmax_err, vmax=vmax_err)

        ax.set_title(title, fontsize=FONT_TITLE, pad=10)
        ax.axis('off')

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=FONT_CB)
        if col == 0:
            cbar.ax.set_title(r'$|u|$', fontsize=FONT_CB_TL, pad=10)

    # 第2行: Temperature
    for col, (data, title) in enumerate([
        (true_T_2d, 'Ground Truth'),
        (saot_T_2d - true_T_2d, 'SAOT Error'),
        (cwgano_T_2d - true_T_2d, 'CWGANO Error')
    ]):
        ax = fig.add_subplot(gs[1, col])

        if col == 0:
            im = ax.imshow(data, cmap='hot', origin='lower')
        else:
            vmax_err = max(abs(data.min()), abs(data.max()))
            im = ax.imshow(data, cmap='RdBu_r', origin='lower', vmin=-vmax_err, vmax=vmax_err)

        if col == 0:
            ax.set_title(title, fontsize=FONT_TITLE, pad=10)
        ax.axis('off')

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=FONT_CB)
        if col == 0:
            cbar.ax.set_title(r'$T$', fontsize=FONT_CB_TL, pad=10)

    plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
    print(f'Saved PDF: {output_path}')

    if png_path:
        plt.savefig(png_path, dpi=150, bbox_inches='tight', format='png')
        print(f'Saved PNG: {png_path}')

    plt.close()
