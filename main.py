import cv2
from image_reader import (
    load_and_preprocess,
    find_biggest_contour,
    warp_perspective,
    split_cells,
    preprocess_cell,
    cell_has_digit,
    extract_digit,
    resize_digit
)
from solver import solve

# Load image
img, thresh = load_and_preprocess(r"C:\Users\Asus\Downloads\sudoku.png")

# Detect grid and warp
grid = find_biggest_contour(thresh)
warped = warp_perspective(img, grid)

# Split into 81 cells
cells = split_cells(warped)

sudoku_matrix = []

for i in range(9):
    row = []
    for j in range(9):
        idx = i * 9 + j
        cell = cells[idx]
        processed = preprocess_cell(cell)

        if cell_has_digit(processed):
            digit_img = extract_digit(processed)
            digit_img = resize_digit(digit_img)

            cv2.imshow("Digit", digit_img)
            cv2.waitKey(1)

            from digit_recognizer import recognize_digit
            num = recognize_digit(digit_img)
            row.append(num)

        else:
            row.append(0)

    sudoku_matrix.append(row)

print("\nDetected Sudoku:")
for r in sudoku_matrix:
    print(r)

# Solve
solve(sudoku_matrix)

print("\nSolved Sudoku:")
for r in sudoku_matrix:
    print(r)

