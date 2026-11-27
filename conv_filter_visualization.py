"""把训练好的卷积网络中的卷积核画成灰度图。

用法：
    python conv_filter_visualization.py
    python conv_filter_visualization.py --model models/mnist_conv_01 --outdir visualizations
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 保存图片，不依赖桌面环境
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from layers_convolution import convolution
from neural_network import network


def collect_conv_layers(net: network) -> list[convolution]:
    return [layer for layer in net.sequence.layers if isinstance(layer, convolution)]


def plot_conv_layer(
    filters: np.ndarray,
    layer_index: int,
    output_path: Path,
    weight_px: int = 5,
    dpi: int = 150,
) -> None:
    """filters 的形状是 (N, C, Hf, Wf)。

    C == 1 时把 N 个卷积核排成 2×5（或接近的网格）；
    C > 1 时按“行 = 输出通道、列 = 输入通道”排成 N×C 的矩阵。
    """
    n, c, hf, wf = filters.shape
    vmax = float(np.abs(filters).max())
    if vmax == 0:
        vmax = 1.0
    norm = Normalize(vmin=-vmax, vmax=vmax)
    cmap = plt.get_cmap("gray")

    if c == 1:
        ncols = min(n, 5)
        nrows = math.ceil(n / ncols)
    else:
        nrows, ncols = n, c

    # 每个卷积核的目标显示尺寸约为 weight_px * 卷积核边长；
    # 12 是留给标题/刻度的额外边距。
    cell_in = (max(hf, wf) * weight_px + 12) / dpi
    left_in = 0.65 if c > 1 else 0.15
    right_in = 0.65
    bottom_in = 0.6 if c > 1 else 0.35
    top_in = 0.55

    fig_w = left_in + ncols * cell_in + right_in
    fig_h = bottom_in + nrows * cell_in + top_in
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(fig_w, fig_h),
        dpi=dpi,
        squeeze=False,
        gridspec_kw={"wspace": 0.10, "hspace": 0.28 if c == 1 else 0.10},
    )
    fig.subplots_adjust(
        left=left_in / fig_w,
        right=1 - right_in / fig_w,
        bottom=bottom_in / fig_h,
        top=1 - top_in / fig_h,
    )

    image = None
    for i in range(nrows):
        for j in range(ncols):
            ax = axes[i][j]
            if c == 1:
                idx = i * ncols + j
                if idx >= n:
                    ax.set_visible(False)
                    continue
                kernel = filters[idx, 0]
            else:
                kernel = filters[i, j]

            image = ax.imshow(
                kernel,
                cmap=cmap,
                norm=norm,
                interpolation="nearest",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            if c == 1:
                ax.set_title(str(idx), fontsize=6, pad=2)
            else:
                if i == 0:
                    ax.set_title(str(j), fontsize=6, pad=2)
                if j == 0:
                    ax.set_ylabel(str(i), fontsize=6, rotation=0, labelpad=8, va="center")

    fig.suptitle(
        f"Conv layer {layer_index}: {n} x {c} x {hf} x {wf}",
        fontsize=8,
        y=0.995,
    )

    if c == 1:
        fig.text(0.5, 0.015, "filter index", ha="center", fontsize=7)
    else:
        fig.text(0.5, 0.012, "input channel", ha="center", fontsize=7)
        fig.text(0.012, 0.5, "output filter", va="center", rotation=90, fontsize=7)

    if image is not None:
        # 0.12in 是色条与滤波网格之间的间距；0.02in 是色条本身宽度。
        cax = fig.add_axes([(fig_w - right_in + 0.12) / fig_w, 0.25, 0.02 / fig_w, 0.5])
        fig.colorbar(image, cax=cax, label="weight")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, facecolor="white")
    plt.close(fig)
    print(f"saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize convolution filters as grayscale images.")
    parser.add_argument("--model", default="models/mnist_conv_01", help="path to a pickled network")
    parser.add_argument("--outdir", default="visualizations", help="directory for the generated PNGs")
    parser.add_argument("--weight-px", type=int, default=5, help="screen pixels per kernel weight")
    parser.add_argument("--dpi", type=int, default=150, help="output image DPI")
    args = parser.parse_args()

    plt.rcParams.update({"font.size": 6})

    net = network.deserialize(args.model)
    conv_layers = collect_conv_layers(net)
    if not conv_layers:
        raise SystemExit(f"no convolution layers found in {args.model}")

    outdir = Path(args.outdir)
    stem = Path(args.model).name
    for index, layer in enumerate(conv_layers, start=1):
        plot_conv_layer(
            layer.filter,
            layer_index=index,
            output_path=outdir / f"{stem}_conv{index}_filters.png",
            weight_px=args.weight_px,
            dpi=args.dpi,
        )


if __name__ == "__main__":
    main()
