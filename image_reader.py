import cv2
import numpy as np

def load_and_preprocess(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    return img, thresh

def find_biggest_contour(thresh):
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    biggest = None
    max_area = 0

    for c in contours:
        area = cv2.contourArea(c)
        if area > max_area:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest

def warp_perspective(img, contour):
    pts = contour.reshape(4,2)

    # order points: top-left, top-right, bottom-right, bottom-left
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    ordered = np.array([tl, tr, br, bl], dtype="float32")

    side = max(
        np.linalg.norm(br - tr),
        np.linalg.norm(tr - tl),
        np.linalg.norm(tl - bl),
        np.linalg.norm(bl - br)
    )

    dst = np.array([
        [0,0],
        [side-1,0],
        [side-1,side-1],
        [0,side-1]
    ], dtype="float32")

    matrix = cv2.getPerspectiveTransform(ordered, dst)
    warped = cv2.warpPerspective(img, matrix, (int(side), int(side)))
    return warped

def split_cells(warped):
    cells = []
    h, w = warped.shape[:2]
    cell_h = h // 9
    cell_w = w // 9

    for i in range(9):
        for j in range(9):
            cell = warped[
                i*cell_h:(i+1)*cell_h,
                j*cell_w:(j+1)*cell_w
            ]
            cells.append(cell)
    return cells
