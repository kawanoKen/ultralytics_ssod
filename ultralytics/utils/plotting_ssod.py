from __future__ import annotations

from pathlib import Path
from typing import Callable

from ultralytics.utils import LOGGER
from ultralytics.utils.plotting import plt_settings


@plt_settings()
def plot_results(file: str = "path/to/results.csv", dir: str = "", on_plot: Callable | None = None):
    """
    SSOD用に拡張した結果可視化。
    - results.csv の列が途中で増減するケース（ragged lines）を許容
    - 軸や凡例の生成が失敗しても安全に終了
    """
    import matplotlib.pyplot as plt  # scoped import
    import polars as pl
    from scipy.ndimage import gaussian_filter1d

    save_dir = Path(file).parent if file else Path(dir)
    files = list(save_dir.glob("results*.csv"))
    assert len(files), f"No results.csv files found in {save_dir.resolve()}, nothing to plot."

    loss_keys, metric_keys = [], []
    columns = []
    fig = None
    ax = None

    for i, f in enumerate(files):
        try:
            # 列不一致を許容（SSOD導入で途中から列が増える）
            data = pl.read_csv(f, infer_schema_length=None, truncate_ragged_lines=True)
            if i == 0:
                for c in data.columns:
                    if "loss" in c:
                        loss_keys.append(c)
                    elif "metric" in c:
                        metric_keys.append(c)
                loss_mid, metric_mid = len(loss_keys) // 2, len(metric_keys) // 2
                columns = loss_keys[:loss_mid] + metric_keys[:metric_mid] + loss_keys[loss_mid:] + metric_keys[metric_mid:]
                if len(columns) == 0:
                    LOGGER.warning("No loss/metric columns found; skipping plot.")
                    continue
                # ncols は ceil にして軸数不足を回避
                import math
                ncols = max(1, math.ceil(len(columns) / 2))
                fig, ax = plt.subplots(2, ncols, figsize=(len(columns) + 2, 6), tight_layout=True)
                ax = ax.ravel()
            # x軸は列名で決定（'epoch' があれば優先、なければ先頭列）
            x_col = "epoch" if "epoch" in data.columns else data.columns[0]
            x = data.select(x_col).to_numpy().flatten()
            for k, col in enumerate(columns):
                # 軸数超過の安全ガード
                if k >= len(ax):
                    break
                try:
                    # 列に欠損が混じるケースでも途中まで描画する
                    import numpy as np
                    try:
                        y = data.select(col).to_numpy().flatten()
                    except Exception:
                        y = data.select(pl.col(col).cast(pl.Float64, strict=False)).to_numpy().flatten()
                    y = y.astype("float", copy=False)
                    # 長さ不一致に備えて切り詰め＋有限値のみを描画
                    n = min(len(x), len(y))
                    xv = x[:n]
                    yv = y[:n]
                    mask = np.isfinite(yv)
                    if not mask.any():
                        continue
                    xv = xv[mask]
                    yv = yv[mask].astype("float")
                except Exception:
                    # 列が存在しない（途中で追加/削除）場合はスキップ
                    continue
                ax[k].plot(xv, yv, marker=".", label=f.stem, linewidth=1.2, markersize=6)
                try:
                    from scipy.ndimage import gaussian_filter1d
                    ax[k].plot(xv, gaussian_filter1d(yv, sigma=3), ":", label="smooth", linewidth=1.0)
                except Exception:
                    pass
                ax[k].set_title(col, fontsize=12)
        except Exception as e:
            LOGGER.error(f"Plotting error for {f}: {e}")

    if fig is not None and ax is not None:
        try:
            if len(ax) > 1:
                ax[1].legend()
            else:
                ax[0].legend()
        except Exception:
            pass
        fname = save_dir / "results.png"
        fig.savefig(fname, dpi=200)
        plt.close()
        if on_plot:
            on_plot(fname)


@plt_settings()
def plot_results_compare(
    files: list[str],
    labels: list[str] | None = None,
    out: str | None = None,
    on_plot: Callable | None = None,
):
    """
    2つ以上の results.csv を受け取り、共通の項目（列）を重ね描きで比較して保存します。
    - CSVごとに列数が異なる（ragged）場合でも共通列のみを描画
    - x軸は各CSVの先頭列（通常 epoch）を使用
    - 凡例は各サブプロットに表示
    """
    import math
    import matplotlib.pyplot as plt  # scoped import
    import polars as pl
    from scipy.ndimage import gaussian_filter1d

    assert files and len(files) >= 2, "files は少なくとも2つ指定してください。"
    if labels and len(labels) != len(files):
        LOGGER.warning("labels の個数が files と一致しません。ファイル名を凡例に使用します。")
        labels = None

    # 読み込み（ragged許容）と共通列抽出
    datas = []
    columns_list = []
    stems = []
    for f in files:
        try:
            df = pl.read_csv(f, infer_schema_length=None, truncate_ragged_lines=True)
            datas.append(df)
            columns_list.append(set(df.columns))
            stems.append(Path(f).stem)
        except Exception as e:
            LOGGER.error(f"Failed to read CSV '{f}': {e}")
            return

    # 共通列（loss/metric系に限定）
    common_cols = set.intersection(*columns_list)
    plotted_cols = sorted([c for c in common_cols if ("loss" in c or "metric" in c)])
    if not plotted_cols:
        LOGGER.warning("共通の loss/metric 列が見つかりません。処理を中止します。")
        return

    # 2行レイアウトで軸を準備
    ncols = max(1, math.ceil(len(plotted_cols) / 2))
    fig, ax = plt.subplots(2, ncols, figsize=(len(plotted_cols) + 2, 6), tight_layout=True)
    ax = ax.ravel() if hasattr(ax, "ravel") else [ax]

    # スタイルをデータセットごとに変えて重なりを視認しやすくする
    line_styles = ["-", "--", ":", "-."]
    alphas = [0.9, 0.9, 0.9, 0.9]

    # 各列について、全CSVを重ね描き
    for i, col in enumerate(plotted_cols):
        if i >= len(ax):
            break
        axis = ax[i]
        for idx, df in enumerate(datas):
            try:
                import numpy as np
                # x軸は列名で決定（'epoch' があれば優先、なければ先頭列）
                x_col = "epoch" if "epoch" in df.columns else df.columns[0]
                x = df.select(x_col).to_numpy().flatten()
                try:
                    y = df.select(col).to_numpy().flatten()
                except Exception:
                    y = df.select(pl.col(col).cast(pl.Float64, strict=False)).to_numpy().flatten()
                # 長さ不一致に備えて切り詰め＋有限値のみを描画（途中までOK）
                n = min(len(x), len(y))
                xv = x[:n]
                yv = y[:n].astype("float", copy=False)
                mask = np.isfinite(yv)
                if not mask.any():
                    continue
                xv = xv[mask]
                yv = yv[mask].astype("float", copy=False)
            except Exception:
                # 列が欠けているCSVはスキップ
                continue
            lab = (labels[idx] if labels else stems[idx])
            ls = line_styles[idx % len(line_styles)]
            alpha = alphas[idx % len(alphas)]
            axis.plot(
                xv,
                yv,
                linestyle=ls,
                linewidth=1.2,
                alpha=alpha,
                label=lab,
                zorder=2 + idx,
            )
            try:
                y_s = gaussian_filter1d(yv, sigma=3)
                axis.plot(
                    xv,
                    y_s,
                    linestyle=ls,
                    linewidth=1.0,
                    alpha=0.6,
                    label=f"{lab} (smooth)",
                    zorder=1 + idx,
                )
            except Exception:
                pass
        axis.set_title(col, fontsize=12)
        axis.legend(fontsize=8, loc="best")

    # 余った軸は非表示
    for j in range(i + 1, len(ax)):
        ax[j].axis("off")

    # 保存
    save_dir = Path(files[0]).parent
    fname = Path(out) if out else (save_dir / "results_compare.png")
    fig.savefig(fname, dpi=200)
    plt.close()
    if on_plot:
        on_plot(fname)


@plt_settings()
def plot_results_map_compare(
    files: list[str],
    labels: list[str] | None = None,
    out: str | None = None,
    on_plot: Callable | None = None,
):
    """
    results.csv 群から mAP 系の共通列のみを抽出して比較描画します。
    - 共通列のうち 'mAP' を含むカラムのみ
    - x軸は 'epoch' があれば優先、なければ先頭列
    - 欠損/長さ不一致は途中まで描画
    """
    import math
    import matplotlib.pyplot as plt  # scoped import
    import polars as pl
    import numpy as np
    from scipy.ndimage import gaussian_filter1d

    assert files and len(files) >= 2, "files は少なくとも2つ指定してください。"
    if labels and len(labels) != len(files):
        LOGGER.warning("labels の個数が files と一致しません。ファイル名を凡例に使用します。")
        labels = None

    datas = []
    columns_list = []
    stems = []
    for f in files:
        try:
            df = pl.read_csv(f, infer_schema_length=None, truncate_ragged_lines=True)
            datas.append(df)
            columns_list.append(set(df.columns))
            stems.append(Path(f).stem)
        except Exception as e:
            LOGGER.error(f"Failed to read CSV '{f}': {e}")
            return

    common_cols = set.intersection(*columns_list)
    plotted_cols = sorted([c for c in common_cols if "mAP" in c])
    if not plotted_cols:
        LOGGER.warning("共通の mAP 列が見つかりません。処理を中止します。")
        return

    ncols = max(1, len(plotted_cols))
    fig, ax = plt.subplots(1, ncols, figsize=(4 * ncols + 2, 3.5), tight_layout=True)
    ax = ax.ravel() if hasattr(ax, "ravel") else [ax]

    line_styles = ["-", "--", ":", "-."]
    markers = ["o", "s", "^", "D", "v", "x", "*", "P"]
    alphas = [0.9, 0.9, 0.9, 0.9]

    for i, col in enumerate(plotted_cols):
        if i >= len(ax):
            break
        axis = ax[i]
        for idx, df in enumerate(datas):
            try:
                x_col = "epoch" if "epoch" in df.columns else df.columns[0]
                x = df.select(x_col).to_numpy().flatten()
                try:
                    y = df.select(col).to_numpy().flatten()
                except Exception:
                    y = df.select(pl.col(col).cast(pl.Float64, strict=False)).to_numpy().flatten()
                n = min(len(x), len(y))
                xv = x[:n]
                yv = np.array(y[:n], dtype=float, copy=False)
                mask = np.isfinite(yv)
                if not mask.any():
                    continue
                xv = xv[mask]
                yv = yv[mask]
            except Exception:
                continue
            lab = (labels[idx] if labels else stems[idx])
            ls = line_styles[idx % len(line_styles)]
            mk = markers[idx % len(markers)]
            alpha = alphas[idx % len(alphas)]
            axis.plot(
                xv,
                yv,
                linestyle=ls,
                marker=mk,
                linewidth=1.2,
                markersize=3,
                alpha=alpha,
                label=lab,
                zorder=2 + idx,
            )
            try:
                y_s = gaussian_filter1d(yv, sigma=3)
                axis.plot(
                    xv,
                    y_s,
                    linestyle=ls,
                    linewidth=1.0,
                    alpha=0.6,
                    label=f"{lab} (smooth)",
                    zorder=1 + idx,
                )
            except Exception:
                pass
        axis.set_title(col, fontsize=12)
        axis.legend(fontsize=8, loc="best")

    for j in range(i + 1, len(ax)):
        ax[j].axis("off")

    save_dir = Path(files[0]).parent
    fname = Path(out) if out else (save_dir / "results_map_compare.png")
    fig.savefig(fname, dpi=200)
    plt.close()
    if on_plot:
        on_plot(fname)


@plt_settings()
def plot_results_da_compare(
    files: list[str],
    labels: list[str] | None = None,
    out: str | None = None,
    on_plot: Callable | None = None,
):
    """
    results.csv 群から DA（domain adaptation）損失系の共通列のみを抽出して比較描画します。
    - 共通列のうち 'da' と 'loss' の両方を含む列（例: loss_da, loss_da_s, loss_da_t, da/loss など）
    - x軸は 'epoch' があれば優先、なければ先頭列
    - 欠損/NaNはスキップして途中まで描画（series丸ごとNaNなら不描画）
    - 点マーカーは表示しない（線のみ、スムージング線あり）
    """
    import math
    import matplotlib.pyplot as plt  # scoped import
    import polars as pl
    import numpy as np
    from scipy.ndimage import gaussian_filter1d

    assert files and len(files) >= 2, "files は少なくとも2つ指定してください。"
    if labels and len(labels) != len(files):
        LOGGER.warning("labels の個数が files と一致しません。ファイル名を凡例に使用します。")
        labels = None

    datas = []
    columns_list = []
    stems = []
    for f in files:
        try:
            df = pl.read_csv(f, infer_schema_length=None, truncate_ragged_lines=True)
            datas.append(df)
            columns_list.append(set(df.columns))
            stems.append(Path(f).stem)
        except Exception as e:
            LOGGER.error(f"Failed to read CSV '{f}': {e}")
            return

    # 全CSVの列の「和集合」からDA損失候補を抽出（片方にしか無い列も描画対象にする）
    union_cols = set().union(*columns_list)
    # DA損失を示す列の抽出（両方含む、ケースインセンシティブ）
    def is_da_loss(col: str) -> bool:
        c = col.lower()
        return ("da" in c and "loss" in c) or c.startswith("da/")

    plotted_cols = sorted([c for c in union_cols if is_da_loss(c)])
    if not plotted_cols:
        LOGGER.warning("共通の DA 損失列が見つかりません。処理を中止します。")
        return

    ncols = max(1, math.ceil(len(plotted_cols)))
    fig, ax = plt.subplots(1, ncols, figsize=(4 * ncols + 2, 3.5), tight_layout=True)
    ax = ax.ravel() if hasattr(ax, "ravel") else [ax]

    line_styles = ["-", "--", ":", "-."]
    alphas = [0.9, 0.9, 0.9, 0.9]

    for i, col in enumerate(plotted_cols):
        if i >= len(ax):
            break
        axis = ax[i]
        for idx, df in enumerate(datas):
            try:
                x_col = "epoch" if "epoch" in df.columns else df.columns[0]
                x = df.select(x_col).to_numpy().flatten()
                try:
                    y = df.select(col).to_numpy().flatten()
                except Exception:
                    y = df.select(pl.col(col).cast(pl.Float64, strict=False)).to_numpy().flatten()
                n = min(len(x), len(y))
                xv = x[:n]
                yv = np.array(y[:n], dtype=float, copy=False)
                mask = np.isfinite(yv)
                if not mask.any():
                    continue
                xv = xv[mask]
                yv = yv[mask]
            except Exception:
                continue
            lab = (labels[idx] if labels else stems[idx])
            ls = line_styles[idx % len(line_styles)]
            alpha = alphas[idx % len(alphas)]
            axis.plot(
                xv,
                yv,
                linestyle=ls,
                linewidth=1.2,
                alpha=alpha,
                label=lab,
                zorder=2 + idx,
            )
            try:
                y_s = gaussian_filter1d(yv, sigma=3)
                axis.plot(
                    xv,
                    y_s,
                    linestyle=ls,
                    linewidth=1.0,
                    alpha=0.6,
                    label=f"{lab} (smooth)",
                    zorder=1 + idx,
                )
            except Exception:
                pass
        axis.set_title(col, fontsize=12)
        axis.legend(fontsize=8, loc="best")

    for j in range(i + 1, len(ax)):
        ax[j].axis("off")

    save_dir = Path(files[0]).parent
    fname = Path(out) if out else (save_dir / "results_da_compare.png")
    fig.savefig(fname, dpi=200)
    plt.close()
    if on_plot:
        on_plot(fname)
