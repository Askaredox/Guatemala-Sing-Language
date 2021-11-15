import tensorflow as tf
import numpy as np
import cv2
import hand_gesture as hg
import matplotlib.pyplot as plt

TAMANO_IMG = 150

class Translator:
    def __init__(self, model=None):

        if(model is None):
            self.modelo = tf.keras.Sequential([
                tf.keras.layers.Dense(units=50, input_shape=(84,), activation = "relu"),
                tf.keras.layers.Dense(units=100, activation = "relu"),
                tf.keras.layers.Dense(units=50, activation = "relu"),
                tf.keras.layers.Dense(units=41, activation = "sigmoid") # ABCDEFGHIJKLMNÑOPQRSTUVWXYZ1234567890
            ])
            self.modelo.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
        else:
            self.modelo = model

    @classmethod
    def from_model(cls, model_path):
        model = tf.keras.models.load_model(model_path)
        return cls(model)

    def save_model(self, path):
        self.modelo.save(path)


    def __train_model(self, input, output):
        print("Comenzando entrenamiento...")
        historial = self.modelo.fit(input, output, epochs=5)
        print("Modelo entrenado!")

        plt.xlabel("# Epoca")
        plt.ylabel("Magnitud de pérdida")
        plt.plot(historial.history["loss"])


    def train(self, path, out):
        print("Comenzando entrenamiento con video "+path)
        input_d = []
        output_d = []
        video = cv2.VideoCapture(path)
        length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        output = self.get_translate(out)
        while True:
            flag, frame = video.read()
            if flag:
                pos_frame = int(video.get(cv2.CAP_PROP_POS_FRAMES))
                image, data = hg.photo_to_data(frame, True)
                cv2.imshow('video', image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                input_d.append(data)
                output_d.append(output)
                print(f'{pos_frame} / {length}', end='\r')
            if(length<=pos_frame):
                break
        
        i_d = np.array(input_d)
        o_d = np.array(output_d)

        self.__train_model(i_d, o_d)

    def get_translate(self, output):
        '''
            _abc¢defghijklłmnñopqr»stuvwxyz1234567890
        '''
        lib = [
            '_', 'A', 'B', 'C', 'CH', 'D', 'E', 'F', 'G', 'H', 'I', 
            'J', 'K', 'L', 'LL', 'M', 'N', 'Ñ', 'O', 'P', 'Q', 
            'R', 'RR', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
            '1', '2', '3', '4', '5', '6', '7', '8', '9', '0'
        ]
        ret = [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        ]

        index = lib.index(output)
        ret[index]=1
        return ret
    
    def prediction(self, input):
        lib = [
            '_', 'A', 'B', 'C', 'CH', 'D', 'E', 'F', 'G', 'H', 'I', 
            'J', 'K', 'L', 'LL', 'M', 'N', 'Ñ', 'O', 'P', 'Q', 
            'R', 'RR', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
            '1', '2', '3', '4', '5', '6', '7', '8', '9', '0'
        ]
        ret = self.modelo.predict(input)[0]
        max_index = np.argmax(ret, axis=0)

        return lib[max_index]

def test_video(model):
    video = cv2.VideoCapture(0)
    while True:
        flag, frame = video.read()
        if flag:
            imagen = cv2.resize(frame, (TAMANO_IMG, TAMANO_IMG))
            imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
            imagen = imagen.reshape(TAMANO_IMG, TAMANO_IMG, 1)
            X = np.array(imagen).astype(float) / 255
            print(model.prediction([X]), end='\r')
            cv2.imshow('MediaPipe Hands', cv2.flip(frame, 1))
            # Flip the image horizontally for a selfie-view display.
            if cv2.waitKey(1) & 0xFF == 27:
                break

def main():
    translate = None
    while True:
        command = input('>').split(' ')
        if(command[0] == 'init'):
            translate = Translator()
        elif(command[0] == 'load'):
            path = command[1]
            translate = Translator.from_model(path)
        elif(command[0] == 'train'):
            path = command[1]
            expect = command[2]
            translate.train(path, expect)
        elif(command[0] == 'save'):
            path = command[1]
            translate.save_model(path)
        elif(command[0] == 'predict'):
            test_video(translate)
        elif(command[0] == 'exit'):
            break

train_data = [
    ['./assets/A/LETRA_A[1].mp4', 'A'],
    ['./assets/A/LETRA_A[2].mp4', 'A'],
    ['./assets/B/LETRA_B[1].mp4', 'B'],
    ['./assets/B/b.mp4', 'B'],
    ['./assets/C/LETRA_C[1].mp4', 'C'],
    ['./assets/D/LETRA_D[1].mp4', 'D'],
    ['./assets/E/e.mp4', 'E'],
    ['./assets/E/LETRA_E[1].mp4', 'E'],
]

if __name__ == "__main__":
    # trans = Translator()
    # for data in train_data:
    #     trans.train(data[0], data[1])
    # trans.save_model('./modelo/modelo_guardado.h5')
    # test_video(trans)
    main()


#train ./assets/A/LETRA_A[1].mp4 A
