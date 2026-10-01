from pathlib import Path
import cv2
import numpy as np

def load_gray(path):
    image = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return image

def extract_candidates(gray):
    threshold, raw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(raw, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda c: (cv2.boundingRect(c)[1], cv2.boundingRect(c)[0]))
    candidates = []
    for candidate_id, contour in enumerate(contours, 1):
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / perimeter**2 if perimeter > 0 else None
        candidates.append(dict(id=candidate_id, area=area, perimeter=perimeter,
                               circularity=circularity, bbox=cv2.boundingRect(contour)))
    return mask, candidates, threshold

def select_candidates(candidates, area_min=900, area_max=3500, circularity_min=0.55):
    return [item for item in candidates if item['perimeter'] > 0
            and area_min <= item['area'] <= area_max
            and item['circularity'] >= circularity_min]

def draw_selected(gray, selected):
    result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for item in selected:
        x, y, w, h = item['bbox']
        cv2.rectangle(result, (x, y), (x+w-1, y+h-1), (40, 190, 30), 1)
        cv2.putText(result, str(item['id']), (x, y+10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.34, (0, 0, 255), 1, cv2.LINE_AA)
    return result
