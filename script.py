import cv2
import numpy as np

def canny(image, blur_ksize, canny_low, canny_high):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, blur_ksize, 0)
    canny_img = cv2.Canny(blur, canny_low, canny_high)
    return canny_img


def region_of_interest(image, mode):
    height = image.shape[0]
    width = image.shape[1]
    mask = np.zeros_like(image)

    if mode == 'road':
        # TRAPEZ pentru drum real
        bottom_limit = int(height * 0.95)
        top_limit = int(height * 0.60)

        polygons = np.array([[
            (int(width * 0.10), bottom_limit),
            (int(width * 0.90), bottom_limit),
            (int(width * 0.55), top_limit),
            (int(width * 0.45), top_limit)
        ]])
    else:
        # Dreptunghiul pentru testul de pe masă
        top_limit = int(height * 0.25)
        bottom_limit = int(height * 0.95)
        left_side = int(width * 0.05)
        right_side = int(width * 0.95)

        polygons = np.array([[
            (left_side, bottom_limit),
            (right_side, bottom_limit),
            (right_side, top_limit),
            (left_side, top_limit)
        ]])

    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


# Algoritm detectie
def process_frame(frame, current_mode):
    # Configurare dimensiune și parametri în funcție de mod
    if current_mode == 'easy':
        proc_frame = cv2.resize(frame, (500, 700))
        blur_ksize = (5, 5)
        canny_low = 50
        canny_high = 150
        hough_thresh = 50
        hough_min_len = 50
        hough_max_gap = 10
        slope_thresh = 0.5

    elif current_mode == 'hard':
        proc_frame = cv2.resize(frame, (500, 700))
        blur_ksize = (15, 15)
        canny_low = 20
        canny_high = 80
        hough_thresh = 20
        hough_min_len = 10
        hough_max_gap = 120
        slope_thresh = 0.3

    elif current_mode == 'road':
        proc_frame = cv2.resize(frame, (1280, 720))
        blur_ksize = (5, 5)
        canny_low = 60
        canny_high = 120
        hough_thresh = 30
        hough_min_len = 20
        hough_max_gap = 200
        slope_thresh_low = 0.4
        slope_thresh_high = 2.5

    canny_image = canny(proc_frame, blur_ksize, canny_low, canny_high)
    cropped_image = region_of_interest(canny_image, current_mode)

    lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, hough_thresh, np.array([]), minLineLength=hough_min_len,
                            maxLineGap=hough_max_gap)

    line_image = np.zeros_like(proc_frame)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            if x1 == x2:
                if current_mode != 'road':
                    cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
                continue

            slope = (y2 - y1) / (x2 - x1)

            if current_mode == 'road':
                if (slope_thresh_low < slope < slope_thresh_high) or (-slope_thresh_high < slope < -slope_thresh_low):
                    cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
            else:
                if slope > slope_thresh or slope < -slope_thresh:
                    cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 3)

    combo_image = cv2.addWeighted(proc_frame, 0.9, line_image, 1, 1)
    cv2.putText(combo_image, f"Mod: {current_mode.upper()}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return combo_image, canny_image, cropped_image


# Meniu Selectie foto/video
print("=========================================")
print("     SISTEM DETECTIE LANE / BENZI        ")
print("=========================================")
print("1. Procesează o FOTOGRAFIE (Imagine statică)")
print("2. Procesează un VIDEOCLIP")
print("=========================================")

optiune = input("Alegeți opțiunea (1 sau 2): ").strip()
current_mode = 'easy'

if optiune == '1':
    cale_foto = input("Introduceți numele pozei (ex: TestR1.jpeg) sau Enter pentru implicit: ").strip()
    
    if not cale_foto:
        cale_foto = './test-images/img_1.png'
    else:
        # Adaugă automat folderul test-images/ în fața numelui pe care l-ai scris
        cale_foto = f'./test-images/{cale_foto}'

    img_statica = cv2.imread(cale_foto)
    if img_statica is None:
        print(f"Eroare: Nu s-a putut încărca imaginea de la calea: {cale_foto}")
        exit()

    print(
        "\n[INFO] Imagine încărcată cu succes. Apăsați tastele 1, 2, 3 pentru a schimba modurile sau ESC pentru a închide.")

    # Rulăm într-o buclă infinită pentru a putea schimba modurile interactiv pe aceeași poză!
    while True:
        combo, canny_img, roi_img = process_frame(img_statica, current_mode)

        cv2.imshow('Result', combo)
        cv2.imshow('Canny', canny_img)
        cv2.imshow("ROI", roi_img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('1'):
            current_mode = 'easy'
        elif key == ord('2'):
            current_mode = 'hard'
        elif key == ord('3'):
            current_mode = 'road'
        elif key == 27:
            break

elif optiune == '2':
    cale_video = input("Introduceți numele videoclipului (ex: test1.mp4) sau Enter pentru implicit: ").strip()
    
    if not cale_video:
        cale_video = './test-images/test1.mp4'
    else:
        # Adaugă automat folderul test-images/ în fața numelui
        cale_video = f'./test-images/{cale_video}'

    cap = cv2.VideoCapture(cale_video)
    if not cap.isOpened():
        print(f"Eroare: Nu s-a putut deschide videoclipul de la calea: {cale_video}")
        exit()

    print("\n[INFO] Video pornit. Apăsați tastele 1, 2, 3 pentru a schimba modurile sau ESC pentru a opri.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Videoclipul s-a terminat cu succes!")
            break

        combo, canny_img, roi_img = process_frame(frame, current_mode)

        cv2.imshow('Result', combo)
        cv2.imshow('Canny', canny_img)
        cv2.imshow("ROI", roi_img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('1'):
            current_mode = 'easy'
        elif key == ord('2'):
            current_mode = 'hard'
        elif key == ord('3'):
            current_mode = 'road'
        elif key == 27:
            break

    cap.release()

else:
    print("Opțiune invalidă! Reporniți programul și alegeți 1 sau 2.")

cv2.destroyAllWindows()