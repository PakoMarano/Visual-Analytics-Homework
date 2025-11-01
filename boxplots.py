import pathlib
import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def load_grades(csv_path: pathlib.Path):
	df = pd.read_csv(csv_path)
	# ensure Grade column exists and numeric
	return pd.to_numeric(df['Grade'], errors='coerce').dropna().tolist()


def plot_boxplots(csv_map, out_path: pathlib.Path):
	# csv_map: list of tuples (label, path)
	grades_data = []
	labels = []
	for label, path in csv_map:
		if not path.exists():
			print(f'Warning: CSV not found: {path}', file=sys.stderr)
			grades = []
		else:
			grades = load_grades(path)
		grades_data.append(grades)
		labels.append(label)

	fig, ax = plt.subplots(figsize=(8, 5))
	# create boxplot; showmeans will show a marker for the mean
	bp = ax.boxplot(grades_data, labels=labels, patch_artist=True, showmeans=True,
					medianprops=dict(color='black'), meanprops=dict(marker='D', markeredgecolor='black', markerfacecolor='firebrick'))

	# color boxes
	colors = ['#8fbfe0', '#f4a261', '#90be6d', '#f94144']
	for patch, color in zip(bp['boxes'], colors):
		patch.set_facecolor(color)

	ax.set_ylabel('Grade')
	ax.set_title('Grades distribution by student (boxplots)')

	# compute overall max to leave space on top so 30 is visible
	all_values = [v for sub in grades_data for v in sub]
	max_val = max(all_values) if all_values else 30
	y_max = max(30, int(max_val)) + 2
	ax.set_ylim(0, y_max)

	# small legend to explain mean marker 
	mean_handle = Line2D([0], [0], marker='D', color='w', markerfacecolor='firebrick', markeredgecolor='black', markersize=6, label='Weighted Mean')
	# place legend at bottom-right
	ax.legend(handles=[mean_handle], loc='lower right', fontsize=8)

	plt.tight_layout(rect=[0, 0, 1, 0.96])
	out_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(out_path, bbox_inches='tight')
	print(f'Boxplots saved to: {out_path}')


def main():
	base = pathlib.Path('data')
	csvs = [
		('Student A', base / 'exams - A.csv'),
		('Student B', base / 'exams - B.csv'),
		('Student C', base / 'exams - C.csv'),
		('Me', base / 'exams - Me.csv'),
	]
	out = pathlib.Path('plots') / 'boxplots.png'
	plot_boxplots(csvs, out)


if __name__ == '__main__':
	main()

