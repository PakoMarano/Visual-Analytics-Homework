import pathlib
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_and_prepare(csv_path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df = df.sort_values('Date').reset_index(drop=True)

    def short_name(s):
        if not isinstance(s, str):
            return str(s)
        return s.split(' - ', 1)[1] if ' - ' in s else s

    def code_of(s):
        if not isinstance(s, str):
            return str(s)
        return s.split(' - ', 1)[0] if ' - ' in s else s[:10]

    df['ShortName'] = df['Exam Name'].apply(short_name)
    df['Code'] = df['Exam Name'].apply(code_of)
    return df


def build_master_index(dfs):
    # dfs: list of DataFrames
    combined = pd.concat(dfs, ignore_index=True)
    # take earliest date per code
    idx = combined.groupby('Code', as_index=False).agg({'Date': 'min', 'ShortName': 'first', 'Weight': 'first'})
    idx = idx.sort_values('Date').reset_index(drop=True)
    return idx


def plot_joint(csv_paths, out_path: pathlib.Path) -> None:
    # Load datasets
    labels = []
    dfs = []
    for label, path in csv_paths:
        df = load_and_prepare(path)
        labels.append(label)
        dfs.append(df)

    # master index of exams ordered by earliest date across datasets
    master = build_master_index(dfs)
    master_codes = master['Code'].tolist()

    # Build a mapping Code -> ShortName (and CFU)
    mapping = {row.Code: f"{row.ShortName} ({int(row.Weight)} CFU)" for row in master.itertuples()}

    # For each dataset, create a series of grades aligned to master_codes
    grade_series = []
    for df in dfs:
        s = df.set_index('Code')['Grade'] if 'Grade' in df.columns else pd.Series(dtype=float)
        s = s.reindex(master_codes)
        grade_series.append(pd.to_numeric(s, errors='coerce'))

    # Compute the per-exam average across available students (ignore NaN)
    stacked = pd.concat(grade_series, axis=1)
    avg_series = stacked.mean(axis=1, skipna=True)

    # Plot
    fig, (ax, ax_map) = plt.subplots(nrows=2, ncols=1, figsize=(14, 6), gridspec_kw={'height_ratios': [5, 1]})

    x = np.arange(len(master_codes))

    colors = ['C0', 'C1', 'C2', 'C3']
    markers = ['o', 's', '^', 'd']

    handles = []
    for i, (label, series) in enumerate(zip(labels, grade_series)):
        h, = ax.plot(x, series.values, marker=markers[i % len(markers)], linestyle='-', color=colors[i % len(colors)], label=label)
        handles.append(h)

    # average line
    h_avg, = ax.plot(x, avg_series.values, marker='x', linestyle='--', color='k', label='Average')
    handles.append(h_avg)

    # formatting
    ax.set_xticks(x)
    ax.set_xticklabels(master_codes, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Grade')
    ax.set_ylim(0, max(30, int(np.nanmax(stacked.values)) + 2))
    ax.set_title('All student grades over time')
    ax.legend(handles=handles, loc='upper left')

    # mapping text on bottom axis (left-justified)
    ax_map.axis('off')
    items = [f"{code}: {mapping[code]}" for code in master_codes]
    # decide columns
    n = len(items)
    if n <= 8:
        ncols = 1
    elif n <= 16:
        ncols = 2
    else:
        ncols = 3

    import math
    rows = math.ceil(n / ncols)
    cols = [items[i * rows:(i + 1) * rows] for i in range(ncols)]
    for c in cols:
        while len(c) < rows:
            c.append('')

    col_widths = [max((len(s) for s in col), default=0) for col in cols]
    lines = []
    for r in range(rows):
        row_parts = []
        for ci, col in enumerate(cols):
            s = col[r]
            row_parts.append(s.ljust(col_widths[ci] + 4))
        lines.append(''.join(row_parts).rstrip())

    mapping_text = '\n'.join(lines)
    ax_map.text(0.01, 0.98, mapping_text, transform=ax_map.transAxes, fontsize=8, va='top', ha='left', family='monospace')

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches='tight')
    print(f'Plot saved to: {out_path}')


def main():
    base = pathlib.Path('data')
    csvs = [
        ('Student A', base / 'exams - A.csv'),
        ('Student B', base / 'exams - B.csv'),
        ('Student C', base / 'exams - C.csv'),
        ('Me', base / 'exams - Me.csv'),
    ]

    for label, p in csvs:
        if not p.exists():
            print(f'Error: CSV not found: {p}', file=sys.stderr)
            sys.exit(2)

    out = pathlib.Path('plots') / 'joint_group_analysis.png'
    plot_joint(csvs, out)


if __name__ == '__main__':
    main()
