from collections import Counter
from pathlib import Path

import cv2
import numpy as np


def candidates(frame, scale=1.0):
    if scale not in (1.0, 0.5):
        raise ValueError('scale must be 1.0 or 0.5')
    h0, w0 = frame.shape[:2]
    small = frame if scale == 1.0 else cv2.resize(
        frame, (w0 // 2, h0 // 2), interpolation=cv2.INTER_AREA)
    sy, sx = small.shape[0] / h0, small.shape[1] / w0
    mask = cv2.inRange(cv2.cvtColor(small, cv2.COLOR_BGR2HSV),
                       (35, 80, 60), (85, 255, 255))
    found = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour) / (sx * sy)
        if area < 80:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        found.append({'xy': ((x + w / 2) / sx, (y + h / 2) / sy),
                      'box': (round(x / sx), round(y / sy),
                              round(w / sx), round(h / sy)),
                      'area': area})
    return sorted(found, key=lambda c: c['xy'][0]), mask


def choose(found, previous, max_distance=35):
    if previous is None:
        return (0, 'start', None) if len(found) == 1 else (None, 'wait', None)
    if not found:
        return None, 'missing', None
    distances = [float(np.hypot(c['xy'][0] - previous[0],
                                c['xy'][1] - previous[1])) for c in found]
    best = int(np.argmin(distances))
    distance = distances[best]
    if sum(abs(d - distance) < 1e-9 for d in distances) > 1:
        return None, 'tie', distance
    if distance > max_distance:
        return None, 'too_far', distance
    return best, 'linked', distance


def overlay(frame, found, chosen=None, trail=()):
    result = frame.copy()
    for i, candidate in enumerate(found):
        x, y, w, h = candidate['box']
        cv2.rectangle(result, (x, y), (x + w - 1, y + h - 1), (0, 220, 255), 2)
        point = tuple(round(v) for v in candidate['xy'])
        cv2.circle(result, point, 4, (0, 220, 255), -1)
        cv2.putText(result, str(i + 1), (x, max(17, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 220, 255), 2)
    if len(trail) > 1:
        cv2.polylines(result, [np.round(trail).astype(np.int32)],
                      False, (255, 100, 20), 3)
    if chosen is not None:
        point = tuple(round(v) for v in found[chosen]['xy'])
        cv2.circle(result, point, 10, (0, 0, 255), 3)
    return result


def run_once(path, clock, fps_from, scale=1.0, display=False):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(path)
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    source_size = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                   int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]
    window = 'Frame processing: close after the run'
    if display:
        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(window, np.zeros((source_size[1], source_size[0], 3), np.uint8))
        cv2.waitKey(1)
    previous, trail, records = None, [], []
    stage = dict.fromkeys(['read', 'detect', 'link', 'draw', 'display'], 0.0)
    completed = 0
    started = clock()
    finished = started
    try:
        while True:
            t0 = clock()
            ok, frame = cap.read()
            t1 = clock()
            if not ok:
                break
            found, _ = candidates(frame, scale)
            t2 = clock()
            chosen, status, distance = choose(found, previous)
            if chosen is None:
                previous, trail = None, []
            else:
                previous = found[chosen]['xy']
                trail.append(previous)
                trail = trail[-30:]
            t3 = clock()
            result = overlay(frame, found, chosen, trail)
            t4 = clock()
            if display:
                cv2.imshow(window, result)
                cv2.waitKey(1)
            t5 = clock()
            for name, dt in zip(stage, (t1-t0, t2-t1, t3-t2, t4-t3,
                                       t5-t4 if display else 0.0)):
                stage[name] += dt
            records.append({'frame': completed, 'candidate_count': len(found),
                            'status': status, 'xy': previous, 'distance': distance})
            completed += 1
            finished = clock()
    finally:
        cap.release()
        if display:
            cv2.destroyWindow(window)
    elapsed = finished - started
    if completed == 0 or elapsed <= 0:
        raise RuntimeError('No completed frames or invalid elapsed time')
    states = Counter(record['status'] for record in records)
    quality = {name: states[name] for name in
               ['start', 'linked', 'missing', 'too_far', 'tie', 'wait']}
    quality['candidate_frames'] = sum(r['candidate_count'] > 0 for r in records)
    quality['selected_frames'] = sum(r['xy'] is not None for r in records)
    return {
        'scale': scale, 'display': display,
        'source_size': source_size,
        'processing_size': [round(v * scale) for v in source_size],
        'source_fps': source_fps, 'completed': completed,
        'elapsed_s': elapsed, 'fps': fps_from(completed, elapsed),
        'mean_frame_ms': elapsed * 1000 / completed,
        'stage_ms': {name: value * 1000 / completed for name, value in stage.items()},
        'other_ms': (elapsed - sum(stage.values())) * 1000 / completed,
        'quality': quality, 'records': records,
    }


def benchmark(path, clock, fps_from, repeats=3, display=True):
    settings = [(1.0, False), (1.0, True), (0.5, False)]
    if not display:
        settings = [(1.0, False), (0.5, False)]
    results = []
    for scale, show in settings:
        run_once(path, clock, fps_from, scale, show)
        for repeat in range(1, repeats + 1):
            result = run_once(path, clock, fps_from, scale, show)
            result['repeat'] = repeat
            results.append(result)
    return results


def print_results(results):
    print('size      display  run  frames  seconds    FPS   ms/frame')
    for r in results:
        size = 'x'.join(map(str, r['processing_size']))
        print(f"{size:9} {str(r['display']):7} {r['repeat']:3} "
              f"{r['completed']:6} {r['elapsed_s']:8.3f} "
              f"{r['fps']:7.1f} {r['mean_frame_ms']:9.3f}")


def plot_results(results):
    import matplotlib.pyplot as plt
    conditions = []
    for r in results:
        key = (r['scale'], r['display'])
        if key not in conditions:
            conditions.append(key)
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, key in enumerate(conditions):
        rows = [r for r in results if (r['scale'], r['display']) == key]
        x = [i + (j - (len(rows)-1)/2) * .16 for j in range(len(rows))]
        ax.scatter(x, [r['mean_frame_ms'] for r in rows], s=75)
    ax.set_xticks(range(len(conditions)), [
        ('576x320' if scale == 1 else '288x160') +
        (' / display on' if show else ' / display off')
        for scale, show in conditions])
    ax.set_ylabel('Mean time per frame (ms)')
    ax.grid(axis='y', alpha=.25)
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def save_preview(path, scale=1.0, output='outputs/preview.avi'):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(path)
    writer = None
    previous, trail = None, []
    frames = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            found, _ = candidates(frame, scale)
            chosen, _, _ = choose(found, previous)
            if chosen is None:
                previous, trail = None, []
            else:
                previous = found[chosen]['xy']
                trail.append(previous)
                trail = trail[-30:]
            result = overlay(frame, found, chosen, trail)
            if writer is None:
                Path(output).parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*'FFV1'),
                                         cap.get(cv2.CAP_PROP_FPS) or 30,
                                         (frame.shape[1], frame.shape[0]))
                if not writer.isOpened():
                    raise RuntimeError('Cannot create preview video')
            writer.write(result)
            frames += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    return frames


def preview_frame(path, scale=1.0, frame_index=80):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(path)
    previous, trail = None, []
    result = None
    try:
        for index in range(frame_index + 1):
            ok, frame = cap.read()
            if not ok:
                raise IndexError(frame_index)
            found, mask = candidates(frame, scale)
            chosen, status, distance = choose(found, previous)
            if chosen is None:
                previous, trail = None, []
            else:
                previous = found[chosen]['xy']
                trail.append(previous)
                trail = trail[-30:]
            result = overlay(frame, found, chosen, trail)
    finally:
        cap.release()
    return {'input': frame, 'mask': mask, 'result': result, 'found': found,
            'chosen': chosen, 'status': status, 'distance': distance}


def compare_frames(path, frame_index=80):
    import matplotlib.pyplot as plt
    previews = [preview_frame(path, scale, frame_index) for scale in (1.0, 0.5)]
    fig, axs = plt.subplots(2, 2, figsize=(12, 7))
    for column, preview in enumerate(previews):
        axs[0, column].imshow(cv2.cvtColor(preview['result'], cv2.COLOR_BGR2RGB))
        mask = cv2.resize(preview['mask'],
                          (preview['input'].shape[1], preview['input'].shape[0]),
                          interpolation=cv2.INTER_NEAREST)
        axs[1, column].imshow(mask, cmap='gray', vmin=0, vmax=255)
        axs[0, column].set_title(('576x320' if column == 0 else '288x160') +
                                f' / frame {frame_index}')
    for ax in axs.flat:
        ax.axis('off')
    fig.tight_layout()
    plt.show()
    plt.close(fig)
    return previews


def print_stages(result):
    print('Stage mean time per frame (ms)')
    for name, value in result['stage_ms'].items():
        print(f'{name:8}: {value:.3f}')
    print(f"{'other':8}: {result['other_ms']:.3f}")
    print(f"{'total':8}: {result['mean_frame_ms']:.3f}")


def print_quality(results):
    print('size      display  run  candidate_frames  linked  selected_frames')
    for r in results:
        size = 'x'.join(map(str, r['processing_size']))
        q = r['quality']
        print(f"{size:9} {str(r['display']):7} {r['repeat']:3} "
              f"{q['candidate_frames']:17} {q['linked']:7} "
              f"{q['selected_frames']:16}")
