import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# For static images:
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
# mp_face = mp.solutions.face


def photo_to_data(frame, debug=False):
    with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5) as hands:

        # print(file)
        # Read an image, flip it around y-axis for correct handedness output (see
        # above).
        image = cv2.flip(frame, 1)
        # Convert the BGR image to RGB before processing.
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Print handedness and draw hand landmarks on the image.
        # print('Hand:', results.multi_handedness)
        data = [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ]

        annotated_image = image.copy()
        if results.multi_hand_landmarks:

            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # print(results)
                # print('hand_landmarks:', hand_landmarks)
                index = results.multi_handedness[i].classification[0].index
                # hand = results.multi_handedness[i].classification[0].label
                for mark, landmark in enumerate(hand_landmarks.landmark):
                    x = landmark.x
                    y = landmark.y
                    data[(index*21+mark)*2] = x
                    data[(index*21+mark)*2+1] = y
                    #print(f'{hand}) hand={index} finger={mark} x={x} y={y}')

                if(debug):
                    mp_drawing.draw_landmarks(
                        annotated_image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
        # print(data)
        return annotated_image, data
        # cv2.imwrite('./annotated_image' + str(idx) + '.png', cv2.flip(annotated_image, 1))

def tests():

# For webcam input:
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        annotated_image, data = photo_to_data(image)

        # Draw the hand annotations on the image.
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        cv2.imshow('MediaPipe Hands', cv2.flip(annotated_image, 1))
        # Flip the image horizontally for a selfie-view display.
        if cv2.waitKey(5) & 0xFF == 27:
            break
    cap.release()
