import cv2
import numpy as np

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (15, 15), 0)

    canny_img = cv2.Canny(blur, 30, 90)
    return canny_img


def region_of_interest(image):
    height = image.shape[0]
    width = image.shape[1]

    top_limit = int(height * 0.25)  # Tăiem top 30%
    bottom_limit = int(height * 0.95)  # Tăiem bottom 5%
    left_side = int(width * 0.05)  # Tăiem 5% din stânga
    right_side = int(width * 0.95)  # Tăiem 5% din dreapta

    polygons = np.array([[
        (left_side, bottom_limit),  # Stânga-jos
        (right_side, bottom_limit),  # Dreapta-jos
        (right_side, top_limit),  # Dreapta-sus
        (left_side, top_limit)  # Stânga-sus
    ]])

    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

cap = cv2.VideoCapture('./test-images/test5.mp4')

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("Videoclipul s-a terminat cu succes!")
        break

    # 1. Micșorăm fereastra pentru a rula fluid și a vedea totul pe ecran
    frame = cv2.resize(frame, (500, 700))

    # 2. Aplicăm detecția de margini și masca (ROI)
    canny_image = canny(frame)
    cropped_image = region_of_interest(canny_image)

    # 3. Detectăm liniile folosind Transformata Hough
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 40, np.array([]), minLineLength=25, maxLineGap=15)

    line_image = np.zeros_like(frame)

    # 4. DESENĂM LINIILE BRUTE (TRECUTE PRIN FILTRUL DE PANTĂ)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x1 == x2:
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
                continue

            slope = (y2 - y1) / (x2 - x1)

            if slope > 0.5 or slope < -0.5:
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 3)

    # 5. Suprapunem liniile detectate peste imaginea originală
    combo_image = cv2.addWeighted(frame, 0.9, line_image, 1, 1)

    # 6. Afișăm rezultatele în ferestre separate pentru diagnoză
    cv2.imshow('Result', combo_image)
    cv2.imshow('Canny', canny_image)
    cv2.imshow("ROI", cropped_image)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()