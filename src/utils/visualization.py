import numpy as np
from kontext.utils import format_time,format_size
import matplotlib.pyplot as plt
import pandas as pd 
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import Normalize


def _fmt_single(metric, value):
    if 'time' in metric.lower():
        return format_time(value).replace('-', '\n')
    if 'memory' in metric.lower() or 'rss' in metric.lower() or 'gpu_peak' in metric.lower():
        return format_size(value)
    return f'{value:.3g}'

def _method_palette(methods):
    return {m: plt.colormaps['tab10'](i % 10) for i, m in enumerate(methods)}

def _x_grid(n_metrics, n_methods, gap=0.9):
    x = np.arange(n_methods)[None, :] + np.arange(n_metrics)[:, None] * (n_methods + gap)
    return x, x.mean(1), x.max() + 1

def _rank_methods_per_sample(sample_scores, metrics, lower_better):
    """
    Convert scores to ranks per (sample, metric). Rank 1 = best.
    NaN scores stay NaN. Ties receive distinct ranks in stable-sort order.
    """
    n_samples, n_metrics, n_methods = sample_scores.shape
    ranks = np.full_like(sample_scores, np.nan)

    for qi, metric in enumerate(metrics):
        reverse = metric in lower_better
        for si in range(n_samples):
            row = sample_scores[si, qi, :]
            finite = np.isfinite(row)
            if not finite.any():
                continue

            values = row[finite]
            order = np.argsort(values, kind='stable')
            method_ranks = np.empty_like(values)

            if reverse:
                # smallest value -> rank 1
                method_ranks[order] = np.arange(1, len(values) + 1)
            else:
                # largest value -> rank 1
                method_ranks[order] = np.arange(len(values), 0, -1)

            ranks[si, qi, finite] = method_ranks
    return ranks

def _rank_dataset_stats(sample_keys, sample_ranks):
    """Mean and std of ranks per dataset."""
    datasets = [k.split('+', 1)[0] for k in sample_keys]
    unique_datasets, seen = [], set()
    for d in datasets:
        if d not in seen:
            seen.add(d)
            unique_datasets.append(d)

    n_d, n_q, n_m = len(unique_datasets), sample_ranks.shape[1], sample_ranks.shape[2]
    mean_ranks = np.full((n_d, n_q, n_m), np.nan)
    std_ranks = np.full((n_d, n_q, n_m), np.nan)

    for di, dataset in enumerate(unique_datasets):
        mask = np.array([d == dataset for d in datasets])
        subset = sample_ranks[mask]
        mean_ranks[di] = np.nanmean(subset, axis=0)
        std_ranks[di] = np.nanstd(subset, axis=0)

    return unique_datasets, mean_ranks, std_ranks

def _collect_methods(results, group_col):
    methods, seen = [], set()
    for records in results.values():
        if len(records)>0:
            for m in pd.DataFrame(records)[group_col]:
                if m not in seen:
                    seen.add(m)
                    methods.append(m)
    return methods

def _aggregate_per_sample(results, group_col, metrics, methods, agg_fn,perfect=True):
    valid_keys = [k for k, r in results.items() if len(r) > 0]

    datasets, seen_datasets = [], set()

    for key in valid_keys:
        dataset = key.split('+', 1)[0]
        if dataset not in seen_datasets:
            seen_datasets.add(dataset)
            datasets.append(dataset)

    sample_keys = []
    for dataset in datasets:
        for key in valid_keys:
            if key.split('+', 1)[0] == dataset:
                sample_keys.append(key)

    n_s, n_q, n_m = len(sample_keys), len(metrics), len(methods)
    scores = np.full((n_s, n_q, n_m), np.nan)

    for si, key in enumerate(sample_keys):
        df = pd.DataFrame(results[key])
        for mi, method in enumerate(methods):
            sub = df[df[group_col] == method]
            for qi, metric in enumerate(metrics):
                vals = sub[metric].dropna().values if metric in sub else []
                if perfect : 
                    if len(vals)==100 :
                        scores[si, qi, mi] = agg_fn(vals)
                else:
                    if len(vals)>0:
                        scores[si, qi, mi] = agg_fn(vals)
                        # print(len(vals),si,qi,mi)

    return sample_keys, scores

def _draw_metric_labels(ax, metrics, group_centres, fontsize):
    for qi, metric in enumerate(metrics):
        ax.text(group_centres[qi], -.25, metric,
                ha='center', va='bottom', fontsize=fontsize + 3,
                fontweight='bold')

def _draw_vertical_gaps(ax, x_positions):
    for qi in range(1, x_positions.shape[0]):
        ax.axvline((x_positions[qi - 1, -1] + x_positions[qi, 0]) / 2,
                   color='grey', ls='--', lw=0.7, alpha=0.45)

def _scores_to_dataframe(sample_keys, tensor, methods, metrics, sep=' | '):
    """tensor: (n_samples, n_metrics, n_methods) → DataFrame (rows=samples,
    cols='method{sep}metric')."""
    n_s, n_q, n_m = tensor.shape
    columns = [f'{methods[mi]}{sep}{metrics[qi]}'
               for qi in range(n_q) for mi in range(n_m)]
    data = tensor.reshape(n_s, n_q * n_m)          # qi outer, mi inner → matches columns
    return pd.DataFrame(data, index=sample_keys, columns=columns)


def readable_dataset_name(dataset):
    return(dataset.replace('ZESTA','Zebrafish embryo').replace('MOSTA','Mouse embryo').replace('ARTISTA','Axolotl').replace('MBA','Mouse brain atlas'))

def readable_sample_name(sample):
    if 'zf' in sample:
        day,section = sample.split('_')
        day = day.replace('zf','hour ')
        section = f'section {section}'
        sample = f'{day}, {section}'
    if 'E1S1' in sample:
        sample = sample.split('_')[0]
        sample = sample.replace('E','day ')
    if 'DPI' in sample:
        if len(sample.split('_')) == 2:
            day,section = sample.replace('DPI','').split('_')
            sample = f'day {day} after injury, section {section}'
        else:
            day = sample.replace('DPI','')
            sample = f'day {day} after injury'

    if 'C57BL6J' in sample:
        section = sample.split('_')[-1]
        sample = f'section {section}'
    return(sample)

def plot_sample_heatmap(
    results,
    group_col='name',
    metrics=('ARI', 'NMI', 'HOM', 'CHAOS_err', 'PAS_err'),
    agg='mean',
    lower_better=('CHAOS_err', 'PAS_err',
                  'embedding_time', 'clustering_time',
                  'embedding_memory_b', 'clustering_memory_b',
                  'loading_time', 'loading_memory_b', 'embedding_time_n',
                  'embedding_peak_rss', 'embedding_delta_rss'),
    cmap_name='RdYlGn',
    fontsize_ticks=10,
    fontsize_scores=6.5,
    rect_width=0.85,
    rect_height=0.45,
    row_spacing=None,
    perfect = True):

    agg_fn = np.median if agg == 'median' else np.mean if agg == 'mean' else np.std if agg =='std' else lambda x: np.mean(x) - np.std(x)
    methods = _collect_methods(results, group_col)
    sample_keys, scores = _aggregate_per_sample(
        results, group_col, metrics, methods, agg_fn,perfect = perfect)

    sample_labels = [readable_sample_name(k.split('+', 1)[1]) for k in sample_keys]
    datasets = [k.split('+', 1)[0] for k in sample_keys]

    n_samples = len(sample_keys)
    n_metrics = len(metrics)
    n_methods = len(methods)

    row_spacing = row_spacing if row_spacing is not None else rect_height + 0.02

    # --- y positions with an extra gap between dataset groups ------------
    group_gap = row_spacing * 1.1
    y = np.empty(n_samples)
    cursor = 0.0
    for si in range(n_samples):
        if si > 0 and datasets[si] != datasets[si - 1]:
            cursor += group_gap
        y[si] = cursor
        cursor += row_spacing

    x_positions, group_centres, x_max = _x_grid(n_metrics, n_methods)

    fig_h = max(y[-1] + rect_height + 1.5, 5)
    fig, ax = plt.subplots(figsize=(max(x_max * 0.72, 10), fig_h))
    best_counts = np.zeros((n_metrics, n_methods), int)

    for qi, metric in enumerate(metrics):
        # print(metric)
        reverse = metric in lower_better
        cmap = plt.cm.get_cmap(cmap_name + ('_r' if reverse else ''))
        for si in range(n_samples):
            row = scores[si, qi, :]
            # print(sample_keys[si],row)
            if not np.isfinite(row).any():
                continue
            lo, hi = np.nanmin(row), np.nanmax(row)
            norm = Normalize(lo, hi) if hi > lo else Normalize(0, 1)
            best = int(np.nanargmin(row) if reverse else np.nanargmax(row))
            best_counts[qi, best] += 1

            for mi, val in enumerate(row):
                cx, cy = x_positions[qi, mi], y[si]
                if np.isnan(val):
                    ax.add_patch(FancyBboxPatch(
                        (cx - rect_width / 2, cy - rect_height / 2),
                        rect_width, rect_height,
                        boxstyle='round,pad=0,rounding_size=0.05',
                        fc='white', ec='#bbbbbb', lw=0.45))
                    ax.text(cx, cy, 'missing', ha='center', va='center',
                            fontsize=fontsize_scores, color='#888888')
                    continue

                is_best = (mi == best)
                ax.add_patch(FancyBboxPatch(
                    (cx - rect_width / 2, cy - rect_height / 2),
                    rect_width, rect_height,
                    boxstyle='round,pad=0,rounding_size=0.05',
                    fc=cmap(norm(val)),
                    ec='black' if is_best else '#bbbbbb',
                    lw=1.8 if is_best else 0.45))
                ax.text(cx, cy, _fmt_single(metric, val),
                        ha='center', va='center', fontsize=fontsize_scores,
                        fontweight='bold' if is_best else 'normal')

    ax.set_yticks(y)
    ax.set_yticklabels(sample_labels, fontsize=fontsize_ticks + 1)

    labels = [f'{methods[mi]} ({best_counts[qi, mi]})'
              for qi in range(n_metrics) for mi in range(n_methods)]
    ax.set_xticks(x_positions.ravel())
    ax.set_xticklabels(labels, fontsize=fontsize_ticks, rotation=50, ha='right')

    for qi, top in enumerate(best_counts.argmax(1)):
        ax.get_xticklabels()[qi * n_methods + top].set_fontweight('bold')

    _draw_metric_labels(ax, metrics, group_centres, fontsize_scores)
    _draw_vertical_gaps(ax, x_positions)

    # --- dataset name above each group + separator lines ----------------
    for si in range(n_samples):
        if si == 0 or datasets[si] != datasets[si - 1]:
            ax.text(-0.55, y[si] - rect_height / 2 - group_gap * 0.05,
                    readable_dataset_name(datasets[si]), ha='left', va='bottom',
                    fontsize=fontsize_ticks + 2, fontweight='bold', color='#333333')

    y_top = y[-1] + rect_height / 2 + 0.2
    y_bottom = -0.9 - rect_height / 2 - group_gap
    ax.set_xlim(-0.65, x_max)
    ax.set_ylim(y_top, y_bottom)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    # --- return scores as DataFrame -------------------------------------
    return fig,ax,_scores_to_dataframe(sample_keys, scores, methods, metrics)

def plot_dataset_rank_dotplot(
    results,
    group_col='name',
    metrics=('ARI', 'NMI', 'HOM', 'COM'),
    agg='mean',
    lower_better=('CHAOS_err', 'PAS_err'),
    fontsize=7,
    dodge=0.10,
    marker_size=45,
    cap_size=2,
    figsize=None,
    perfect=True):
    """
    Cleveland dot plot of method ranks: one panel per metric.
    Returns a DataFrame (rows = samples, cols = 'method | metric')
    holding the per-sample mean rank of each method (1 = best).
    """
    


    agg_fn = np.median if agg == 'median' else np.mean
    methods = _collect_methods(results, group_col)
    sample_keys, sample_scores = _aggregate_per_sample(results, group_col, metrics, methods, agg_fn,perfect = perfect)
    sample_ranks = _rank_methods_per_sample(sample_scores, metrics, lower_better)
    unique_datasets, mean_ranks, std_ranks = _rank_dataset_stats(sample_keys, sample_ranks)

    colors = _method_palette(methods)
    n_datasets = len(unique_datasets)
    n_metrics = len(metrics)
    n_methods = len(methods)

    fig, axes = plt.subplots(1, n_metrics,
                             figsize=figsize or (2.8 * n_metrics, 4.0),
                             sharex=False)
    axes = np.atleast_1d(axes)

    for qi, metric in enumerate(metrics):
        ax = axes[qi]
        for mi, method in enumerate(methods):
            ranks = mean_ranks[:, qi, mi]
            err = std_ranks[:, qi, mi]
            y = np.arange(n_datasets) + mi * dodge - (n_methods - 1) * dodge / 2
            valid = np.isfinite(ranks)

            ax.errorbar(ranks[valid], y[valid], xerr=err[valid],
                        fmt='o', markersize=np.sqrt(marker_size),
                        capsize=cap_size, color=colors[method],
                        ecolor=colors[method],
                        label=method if qi == 0 else '',
                        linewidth=0.8, markeredgecolor='black',
                        markeredgewidth=0.5)
        ax.set_yticks(np.arange(n_datasets))
        if qi == 0:
            ax.set_yticklabels(unique_datasets, fontsize=fontsize)
        else:
            ax.set_yticklabels([])
        ax.set_title(metric, fontsize=fontsize + 2, fontweight='bold')
        ax.set_xlabel('Mean rank (1 = best)', fontsize=fontsize)
        ax.set_xlim(0.5, n_methods + 0.5)
        ax.set_xticks(range(1, n_methods + 1))
        ax.tick_params(axis='x', labelsize=fontsize)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='x', linestyle='--', linewidth=0.5, alpha=0.35)
        ax.set_axisbelow(True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=n_methods,
               fontsize=fontsize, frameon=False,
               bbox_to_anchor=(0.5, 0.12))


    # --- return per-sample ranks as DataFrame ---------------------------
    return fig,axes,_scores_to_dataframe(sample_keys, sample_ranks, methods, metrics)

def plot_metric_vs_nobs(
    results,
    benchmark,                                   # list of (dataset, sample) pairs
    metric,
    group_col = 'name',
    agg       = 'mean',                          # per-sample aggregation
    ax        = None,
    logx      = True,
    logy      = False,):
    """Scalability plot: aggregated `metric` vs n_obs, one line per method.
    Method order (and thus colours) follow first appearance in the data,
    so they stay consistent across plots.
    y-ticks are formatted with format_time / format_size when relevant."""
    from utils.load_data import get_dataset_info

    aggf      = np.median if agg == 'median' else np.mean
    n_obs_of  = lambda ds, sn: get_dataset_info()[ds][sn]['n_obs']
    formatter = ((lambda x: format_time(x).replace('-', '\n')) if 'time'   in metric.lower() else
                 format_size if 'memory' in metric.lower() or 'rss' in metric.lower() else None)

    # fixed method order + colour map (shared across plots) ---------------
    methods = _collect_methods(results, group_col)
    colors  = _method_palette(methods)

    # ---- 1. long table: one (n_obs, method, value) row per sample --------
    rows = []
    for ds, sn in benchmark:
        if not results.get(f'{ds}+{sn}'):
            continue
        df = pd.DataFrame(results[f'{ds}+{sn}'])
        if metric not in df:
            continue
        for m, sub in df.groupby(group_col):
            v = sub[metric].dropna().values

            if len(v)==100:
                rows.append({'n_obs': n_obs_of(ds, sn), group_col: m, 'val': aggf(v)})

    long = pd.DataFrame(rows)
    print(long.values)
    # ---- 2. one line per method (averaged over samples sharing n_obs) -----
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4.5))
    for m in methods:                                    # fixed order
        sub = long[long[group_col] == m]
        if sub.empty:
            continue
        s = sub.groupby('n_obs')['val'].mean().sort_index()
        ax.plot(s.index, s.values, marker='o', label=m, color=colors[m])

    # ---- 3. decorations --------------------------------------------------
    if logx: ax.set_xscale('log')
    if logy: ax.set_yscale('log')
    if formatter:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: formatter(v)))
    ax.set_xlabel('n_obs') 
    ax.set_ylabel(metric)
    ax.set_title(metric)
    ax.legend(fontsize=15, frameon=False)
    ax.figure.tight_layout()
    return ax
