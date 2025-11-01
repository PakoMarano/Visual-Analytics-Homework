import pathlib
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_and_prepare(csv_path) -> pd.DataFrame:
	df = pd.read_csv(csv_path)
	# Parse dates
	df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
	# Order by date
	df = df.sort_values('Date').reset_index(drop=True)
	
	# Extract a compact code
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
	
	# Use compact Code as tick labels
	df['Label'] = df['Code']
	return df


def plot_grades(df, out_path) -> None:
	grades = df['Grade'].astype(float)
	weights = df['Weight'].astype(float)
	arithmetic = grades.mean()
	weighted = np.average(grades, weights=weights)

	n = len(df)
	x = np.arange(n)

	# Create a figure with two rows: main plot on top, mapping/legend area below
	fig, (ax, ax_legend) = plt.subplots(nrows=2, ncols=1, figsize=(max(10, n * 0.6), 6), gridspec_kw={'height_ratios': [5, 1]})
	ax.plot(x, grades, marker='o', linestyle='-', label='Grade')
	ax.scatter(x, grades)

	# Horizontal lines for means
	ax.hlines(arithmetic, xmin=-0.5, xmax=n - 0.5, colors='C1', linestyles='--', label=f'Arithmetic mean: {arithmetic:.2f}')
	ax.hlines(weighted, xmin=-0.5, xmax=n - 0.5, colors='C2', linestyles='-.', label=f'Weighted mean: {weighted:.2f}')

	# Annotate each point with the grade value
	for xi, g in zip(x, grades):
		ax.annotate(f'{g:.0f}', (xi, g), textcoords='offset points', xytext=(0, 6), ha='center', fontsize=8)

	ax.set_xticks(x)
	# Use compact codes as X labels to improve readability
	ax.set_xticklabels(df['Label'], rotation=45, ha='right', fontsize=8)
	ax.set_xlabel('Exam, ordered by date')
	ax.set_ylabel('Grade')
	ax.set_title('My exam grades over time')
	ax.set_ylim(0, max(30, grades.max() + 2))
	ax.legend()
	# Build a legend-like mapping of Code -> ShortName
	# Include CFU in the mapping
	legend_lines = [f"{row.Code}: {row.ShortName} ({int(row.Weight)} CFU)" for row in df.itertuples()]
	legend_text = '\n'.join(legend_lines)
	# Put the mapping into the dedicated bottom axes
	ax_legend.axis('off')
	# Left-justify the mapping text
	ax_legend.text(0.01, 0.98, legend_text, transform=ax_legend.transAxes, fontsize=8, va='top', ha='left', family='monospace')
	plt.tight_layout()
	fig.savefig(out_path, bbox_inches='tight')
	print(f'Plot saved to: {out_path}')


def main(csv_path, out_path):
	try:
		csv_path = pathlib.Path(csv_path)
	except Exception as e:
		print(f'Error: invalid CSV path: {e}', file=sys.stderr)
		sys.exit(2)
	df = load_and_prepare(csv_path)
	plot_grades(df, out_path)


if __name__ == '__main__':
	main('data/exams - Me.csv', 'plots/individual_analysis.png')

