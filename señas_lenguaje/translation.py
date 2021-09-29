import tensorflow as tf
import numpy as np
import csv
import hand_gesture as hg
import matplotlib.pyplot as plt

path = './images/'
inputs = []
outputs = []

with open(f'{path}database.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
    for i, row in enumerate(spamreader):
        if i != 0:
            filename = ''
            d_output = []
            for i, data in enumerate(row):
                if i == 0: filename = data
                else: d_output.append(data)
            d_input = hg.photo_to_data(path + filename)

            i = np.array(d_input, dtype=float)
            inputs.append(i)
            o = np.array(d_output, dtype=float)
            outputs.append(o)
            print(f'{filename} analized!!')

inputs = np.array(inputs)
outputs = np.array(outputs)


oculta1 = tf.keras.layers.Dense(units=50, input_shape=[84])
oculta2 = tf.keras.layers.Dense(units=100)
oculta2 = tf.keras.layers.Dense(units=50)
salida = tf.keras.layers.Dense(units=40)
modelo = tf.keras.Sequential([oculta1, oculta2, salida])

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)

print("Comenzando entrenamiento...")
historial = modelo.fit(inputs, outputs, epochs=1000)
print("Modelo entrenado!")

plt.xlabel("# Epoca")
plt.ylabel("Magnitud de pérdida")
plt.plot(historial.history["loss"])


print("Variables internas del modelo")
#print(capa.get_weights())
print(oculta1.get_weights())
print(oculta2.get_weights())
print(salida.get_weights())

modelo.save('./model/sign_lang.h5')

'''
abc¢defghijklłmnñopqr»stuvwxyz1234567890
'''