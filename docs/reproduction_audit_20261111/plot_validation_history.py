"""Plot the recovered validation history; does not load a model or dataset."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
with (root / 'validation_history.csv').open() as stream:
    rows = list(csv.DictReader(stream))
epochs = [int(r['epoch']) for r in rows]
values = [float(r['validation_step_loss']) for r in rows]
assert epochs == list(range(65))
best = min(range(len(rows)), key=lambda i: values[i])
assert epochs[best] == 59
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(epochs, values, color='#286aa6', marker='.', markersize=4, linewidth=1.5)
ax.scatter([epochs[best]], [values[best]], color='#b83a2b', marker='*', s=150, zorder=5)
ax.annotate(f'Best: epoch 59\nloss {values[best]:.8f}', (epochs[best], values[best]),
            xytext=(-125, 35), textcoords='offset points',
            arrowprops={'arrowstyle': '->', 'color': '#b83a2b'})
ax.axvline(64, color='#777777', linestyle='--', linewidth=1, label='Early stop: epoch 64')
ax.set(xlabel='Epoch index (zero based)', ylabel='validation_step_loss',
       title='Epoch 59 selection from the recovered W&B history\nCombined training objective; not MPJPE or mesh error')
ax.grid(alpha=.2)
ax.legend()
fig.tight_layout()
for extension in ['png', 'pdf']:
    fig.savefig(root / f'validation_curve.{extension}', dpi=170, bbox_inches='tight')
plt.close(fig)
