import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
# mp_face = mp.solutions.face


def photo_to_data(file):
    with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5) as hands:
        
        #print(file)
        # Read an image, flip it around y-axis for correct handedness output (see
        # above).
        image = cv2.flip(cv2.imread(file), 1)
        # Convert the BGR image to RGB before processing.
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Print handedness and draw hand landmarks on the image.
        # print('Hand:', results.multi_handedness)
        data = [
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        ]
        if not results.multi_hand_landmarks:
            return data
        annotated_image = image.copy()
        
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # print(results)
            # print('hand_landmarks:', hand_landmarks)
            index = results.multi_handedness[i].classification[0].index
            # hand = results.multi_handedness[i].classification[0].label
            for mark, landmark in enumerate(hand_landmarks.landmark):
                x = landmark.x
                y = landmark.y
                data[(index*21+mark)*2]=x
                data[(index*21+mark)*2+1]=y
                #print(f'{hand}) hand={index} finger={mark} x={x} y={y}')

            mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
        return data
        #print(data)
        # cv2.imwrite('./annotated_image' + str(idx) + '.png', cv2.flip(annotated_image, 1))
