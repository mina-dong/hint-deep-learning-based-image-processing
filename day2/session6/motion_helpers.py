from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

def read_image(path):
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image

def gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def show(images, titles):
    fig, axes = plt.subplots(1, len(images), figsize=(5*len(images), 4), squeeze=False)
    for ax, image, title in zip(axes[0], images, titles):
        if image.ndim == 2:
            ax.imshow(image, cmap='gray', vmin=0, vmax=255)
        else:
            ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    plt.close(fig)

def mark_changes(frame, mask):
    clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result, boxes = frame.copy(), []
    for contour in contours:
        if cv2.contourArea(contour) < 30:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        boxes.append((x,y,w,h))
        cv2.rectangle(result, (x,y), (x+w-1,y+h-1), (0,0,255), 1)
    return clean, result, boxes

def process_video(path, difference_mask, threshold_value=20):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError('Cannot open video')
    output = Path('outputs')
    output.mkdir(exist_ok=True)
    writer = cv2.VideoWriter(str(output/'changes.avi'), cv2.VideoWriter_fourcc(*'FFV1'), 10, (660,210))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError('Cannot create video')
    prev_gray, index, counts, samples = None, 0, [], []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            current = gray(frame)
            if prev_gray is not None:
                diff, mask = difference_mask(current, prev_gray, threshold_value)
                counts.append(cv2.countNonZero(mask))
                panel = np.hstack([frame, cv2.cvtColor(diff,cv2.COLOR_GRAY2BGR), cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)])
                writer.write(panel)
                if index in (50,60,110):
                    samples.append(panel)
                    cv2.imencode('.png', panel)[1].tofile(str(output/f'frame_{index+350}.png'))
            prev_gray = current.copy()
            index += 1
    finally:
        cap.release()
        writer.release()
    for image, number in zip(samples,(400,410,460)):
        show([image], [f'Source frame {number}: input / difference / mask'])
    return index, counts

def run_camera(difference_mask, threshold_value=20, camera_id=0):
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError('Cannot open camera')
    prev_gray = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            current = gray(frame)
            if prev_gray is not None and current.shape == prev_gray.shape:
                diff, mask = difference_mask(current, prev_gray, threshold_value)
                cv2.imshow('Input',frame)
                cv2.imshow('Difference',diff)
                cv2.imshow('Mask',mask)
                if cv2.waitKey(1) & 0xff == ord('q'):
                    break
            prev_gray = current.copy()
    finally:
        cap.release()
        cv2.destroyAllWindows()
