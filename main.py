import cv2
import mediapipe as mp
import numpy as np

#  MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Функция для размывания области вокруг конкретных координат
def blur_around_point(image, point, radius=30):
    x, y = point
    x, y = int(x), int(y)

    blurred_image = cv2.medianBlur(image, 41)  # медианное размытие
    output = image.copy()

    x_start = max(x - radius, 0)
    x_end = min(x + radius, image.shape[1])
    y_start = max(y - radius, 0)
    y_end = min(y + radius, image.shape[0])


    output[y_start:y_end, x_start:x_end] = blurred_image[y_start:y_end, x_start:x_end]

    return output


def is_inappropriate_gesture(hand_landmarks):
    # Для каждой руки извлекаем ключевые точки
    h, w, _ = image.shape
    keypoints = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]


    thumb_tip = keypoints[4]  # Кончик большого пальца
    index_tip = keypoints[8]  # Кончик указательного пальца
    middle_tip = keypoints[12]  # Кончик среднего пальца
    ring_tip = keypoints[16]  # Кончик безымянного пальца
    pinky_tip = keypoints[20]  # Кончик мизинца

    # Проверяем простое условие: если кончик среднего пальца выше кончика большого пальца
    if middle_tip[1] < thumb_tip[1] and middle_tip[0] > thumb_tip[0]:
        return True
    
    return False

# Захват видео с веб-камеры
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Не удалось получить кадр.")
        continue

    image = cv2.flip(image, 1)
    # Преобразование изображения в RGB для MediaPipe
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Обработка изображения и извлечение ключевых точек
    results = hands.process(image_rgb)

    # Если найдены руки
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            if is_inappropriate_gesture(hand_landmarks):
                for lm in hand_landmarks.landmark:
                    h, w, _ = image.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    image = blur_around_point(image, (cx, cy), radius=30)

    cv2.imshow('Blurred Areas in Real-Time', image)

    if cv2.waitKey(5) & 0xFF == 27:  # Нажмите ESC для выхода
        break

cap.release()
cv2.destroyAllWindows()