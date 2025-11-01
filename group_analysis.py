import pathlib
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_and_prepare(csv_path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Parse dates and order by date
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df = df.sort_values('Date').reset_index(drop=True)

    def short_name(s) -> str:
        if not isinstance(s, str):
            return str(s)
        return s.split(' - ', 1)[1] if ' - ' in s else s

    def code_of(s) -> str:
        if not isinstance(s, str):
            return str(s)
        return s.split(' - ', 1)[0] if ' - ' in s else s[:10]

    df['ShortName'] = df['Exam Name'].apply(short_name)
    df['Code'] = df['Exam Name'].apply(code_of)
    df['Label'] = df['Code']
    return df


def plot_group(csv_paths, out_path) -> None:
    # csv_paths: list of (label, path)
    data = []
    for label, path in csv_paths:
        df = load_and_prepare(path)
        data.append((label, df))

    n_plots = len(data)
    # 2x2 layout for 4 datasets
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axs = axs.flatten()

    # We'll collect a representative line for the legend
    legend_handles = []
    legend_labels = []

    for ax, (label, df) in zip(axs, data):
        grades = df['Grade'].astype(float)
        weights = df['Weight'].astype(float)
        arithmetic = grades.mean()
        weighted = np.average(grades, weights=weights)

        x = np.arange(len(df))

        # main grade line
        h_line, = ax.plot(x, grades, marker='o', linestyle='-', label='Grade')
        ax.scatter(x, grades)

        # means
        h_arith = ax.hlines(arithmetic, xmin=-0.5, xmax=len(df) - 0.5, colors='C1', linestyles='--', label='Arithmetic mean')
        h_weight = ax.hlines(weighted, xmin=-0.5, xmax=len(df) - 0.5, colors='C2', linestyles='-.', label='Weighted mean')

        # annotate points
        for xi, g in zip(x, grades):
            ax.annotate(f'{g:.0f}', (xi, g), textcoords='offset points', xytext=(0, 6), ha='center', fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(df['Label'], rotation=45, ha='right', fontsize=8)
        ax.set_ylim(0, max(30, grades.max() + 2))
        ax.set_title(label)
        ax.set_ylabel('Grade')

        # Only collect legend handles once (from the first plotted axes)
        if not legend_handles:
            legend_handles = [h_line, h_arith, h_weight]
            legend_labels = ['Grade', 'Arithmetic mean', 'Weighted mean']

    # Place a single legend below all subplots
    fig.legend(legend_handles, legend_labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.02))

    # Build a mapping of Code -> ShortName (with CFU) across all datasets (preserve first occurrence)
    mapping = {}
    for _, df in data:
        for row in df.itertuples():
            code = getattr(row, 'Code')
            if code not in mapping:
                mapping[code] = f"{getattr(row, 'ShortName')} ({int(getattr(row, 'Weight'))} CFU)"

    # Format mapping into 2 columns
    items = [f"{code}: {name}" for code, name in mapping.items()]
    n = len(items)
    ncols = 2
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

    # Create a dedicated bottom axis for the mapping so it never overlaps the plots
    # Use figure coordinates: [left, bottom, width, height]
    ax_map = fig.add_axes([0.06, 0.02, 0.88, 0.16])
    ax_map.axis('off')
    ax_map.text(0.0, 1.0, mapping_text, transform=ax_map.transAxes, fontsize=8, va='top', ha='left', family='monospace')

    # Adjust main layout to leave space for the mapping axis
    plt.suptitle('Exam grades over time by student')
    plt.tight_layout(rect=[0, 0.18, 1, 0.96])
    fig.savefig(out_path, bbox_inches='tight')
    print(f'Plot saved to: {out_path}')


def main():
    # default CSV files in the data/ folder
    base = pathlib.Path('data')
    csvs = [
        ('Student A', base / 'exams - A.csv'),
        ('Student B', base / 'exams - B.csv'),
        ('Student C', base / 'exams - C.csv'),
        ('Me', base / 'exams - Me.csv'),
    ]

    # Validate files
    for label, p in csvs:
        if not p.exists():
            print(f'Error: CSV not found: {p}', file=sys.stderr)
            sys.exit(2)

    out = pathlib.Path('plots')
    out.mkdir(parents=True, exist_ok=True)
    plot_group(csvs, out / 'group_analysis.png')


if __name__ == '__main__':
    main()
