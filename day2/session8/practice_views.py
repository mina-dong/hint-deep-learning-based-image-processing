from pathlib import Path
import csv

import cv2
import matplotlib.pyplot as plt


def plot_stages(result):
    values = dict(result['stage_ms'])
    values['other'] = result['other_ms']
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.barh(list(values), list(values.values()), color='#1768AC')
    ax.set_xlabel('Mean time per frame (ms)')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=.2)
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def show_crop_comparison(previews, crop=(227, 35, 357, 165)):
    x0, y0, x1, y1 = crop
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    for column, preview in enumerate(previews):
        result = cv2.cvtColor(preview['result'], cv2.COLOR_BGR2RGB)
        mask = cv2.resize(preview['mask'],
                          (preview['input'].shape[1], preview['input'].shape[0]),
                          interpolation=cv2.INTER_NEAREST)
        axs[0, column].imshow(result[y0:y1, x0:x1])
        axs[1, column].imshow(mask[y0:y1, x0:x1], cmap='gray', vmin=0, vmax=255)
        axs[0, column].set_title('576x320' if column == 0 else '288x160')
    for ax in axs.flat:
        ax.axis('off')
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def save_measurements(results, path='outputs/measurements.csv'):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for result in results:
        row = {key: result[key] for key in
               ['scale', 'display', 'repeat', 'completed', 'elapsed_s',
                'fps', 'mean_frame_ms', 'source_fps']}
        row.update({key + '_ms': value for key, value in result['stage_ms'].items()})
        row['other_ms'] = result['other_ms']
        row.update(result['quality'])
        rows.append(row)
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def plot_fps(results):
    rows = sorted(results, key=lambda r: r['repeat'])
    fig, ax = plt.subplots(figsize=(8, 3.5))
    x = [r['repeat'] for r in rows]
    values = [r['fps'] for r in rows]
    bars = ax.bar(x, values, color='#2563EB')
    ax.set_xticks(x, [str(i) for i in x])
    ax.set_xlabel('Run')
    ax.set_ylabel('Processing FPS')
    ax.set_ylim(0, max(values) * 1.25)
    ax.bar_label(bars, fmt='%.1f', padding=4)
    ax.grid(axis='y', alpha=.2)
    ax.set_axisbelow(True)
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def show_sizes(results):
    rows = []
    for scale in (1.0, 0.5):
        runs = [r for r in results if r['scale'] == scale and not r['display']]
        if len(runs) != 3:
            raise ValueError('Three display-off runs are required for each size')
        rows.append(sorted(runs, key=lambda r: r['elapsed_s'])[1])
    print('Size      Run  Detect(ms/frame)  Total(ms/frame)  Processing FPS')
    for r in rows:
        size = 'x'.join(map(str, r['processing_size']))
        print(f"{size:9} {r['repeat']:3} {r['stage_ms']['detect']:17.3f} "
              f"{r['mean_frame_ms']:16.3f} {r['fps']:15.1f}")
    return rows
