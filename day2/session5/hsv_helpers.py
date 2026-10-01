from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

def read_image(path):
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image

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

def candidates(frame, mask, min_area=300):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result, boxes = frame.copy(), []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        cx, cy = x+w/2, y+h/2
        boxes.append((x, y, w, h, cx, cy))
        cv2.rectangle(result, (x,y), (x+w-1,y+h-1), (0,0,255), 2)
        cv2.circle(result, (int(cx),int(cy)), 3, (0,0,255), -1)
    return result, boxes

def compare(frame, color_mask, bounds):
    masks = [color_mask(frame, lo, hi) for lo, hi in bounds]
    counts = [cv2.countNonZero(mask) for mask in masks]
    show(masks, [str(n)+' pixels' for n in counts])
    return counts

def video_samples(path, color_mask, lower, upper):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError('Cannot open video')
    images, boxes, index = [], [], 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            mask = color_mask(frame, lower, upper)
            result, found = candidates(frame, mask)
            if index in (10, 60, 160):
                images.append(result)
                boxes.append(found)
            index += 1
    finally:
        cap.release()
    show(images, ['Source frame 50', 'Source frame 100', 'Source frame 200'])
    return index, boxes

def run_camera(color_mask, lower, upper, camera_id=0):
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError('Cannot open camera')
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            mask = color_mask(frame, lower, upper)
            result, _ = candidates(frame, mask)
            cv2.imshow('HSV candidates - q to quit', result)
            if cv2.waitKey(1) & 0xff == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
