import cv2
import matplotlib.pyplot as plt
import torch
import pickle
import numpy as np

cap = cv2.VideoCapture('.\VideoBus\_43.avi')#загружаем видео для тренировочной выборки

x_train = []#здесь содержатся тренировочные изображения, уже преобразованные через resharp
y_train = []#здесь содержатся маркеры к фотографиям (1 - мало, 2 - норма, 3 - много)
x_validation = []#валидационные данные
y_validation = []#валидационные ответы
weigth = 3#значение веса кадра
i = 0#счётчик

#тренировочная выборка

#обработка первого видео
while True:
    ret, frame = cap.read()
    if ret == True:
        img = cv2.resize(frame, (72, 128))#уменьшаем изображение в 100 раз 1280х720 -> 128х72
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#удаляем лишние цветовые каналы

        if i == 1125:
            weigth = 2
        if i == 7850:
            weigth = 1
        if i % 10 == 0:
            x = torch.from_numpy(img_gray)#получаем tensor из массива Numpy
            x = x.float()#преобразование типов
            x = x.reshape([72*128])#делаем одномерный tensor
            x_train.append(x)
            y_train.append(weigth)
        i += 1

    elif i > 19500:
        break
cap.release()#освобождаем камеру
print("1 done " + str(len(x_train)))#лог того, что первое видео обработалось

#обработка второго видео
cap = cv2.VideoCapture('.\VideoBus\_7.avi')#загружаем видео для тренировочной выборки

i = 0#счётчик
weigth = 3

while True:
    ret, frame = cap.read()
    if ret == True:
        img = cv2.resize(frame, (72, 128))#уменьшаем изображение в 100 раз 1280х720 -> 128х72
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#удаляем лишние цветовые каналы

        if i == 7050:
            weigth = 2
        if i == 12300:
            weigth = 1
        if i % 10 == 0:
            x = torch.from_numpy(img_gray)
            x = x.float()
            x = x.reshape([72 * 128])
            x_train.append(x)
            y_train.append(weigth)
        i += 1
    elif i > 19500:
        break
cap.release()#освобождаем камеру
print("2 done " + str(len(x_train)))#лог

#обработка третьего видео
cap = cv2.VideoCapture('.\VideoBus\_40.avi')#загружаем видео

i = 0#счётчик
weigth = 3

while True:
    ret, frame = cap.read()
    if ret == True:
        img = cv2.resize(frame, (72, 128))#1280x720 -> 128x72
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#удаление лишних цветовых каналов

        if i == 14550:
            weigth = 2
        if i % 10 == 0:
            x = torch.from_numpy(img_gray)
            x = x.float()
            x = x.reshape([72 * 128])
            x_train.append(x)
            y_train.append(weigth)
        i += 1
    elif i > 19500:
        break
cap.release()#освобождаем камеру
print("3 done " + str(len(x_train)))#лог

#запись тренировочной выборки в файл
f = open(r'x_train.txt', 'wb')
pickle.dump(x_train, f)
f.close()

f = open(r'y_train.txt', 'wb')
pickle.dump(y_train, f)
f.close()

#очищаем списки, для экономии оперативной памяти
x_train.clear()
y_train.clear()

#валидационные выборки
cap = cv2.VideoCapture('.\VideoBus\_3.avi')#загружаем видео

i = 0#счётчик
weigth = 1

while True:
    ret, frame = cap.read()
    if ret == True:
        img = cv2.resize(frame, (72, 128))#1280x720 -> 128x72
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#удаляем лиишние цветовые каналы

        if i % 10 == 0:
            x = torch.from_numpy(img_gray)
            x = x.float()
            x = x.reshape([72 * 128])
            x_validation.append(x)
            y_validation.append(weigth)
        i += 1
    elif i > 19500:
        break
cap.release()#освобождаем камеру


cap = cv2.VideoCapture('.\VideoBus\_16.avi')#загружаем видео

i = 0#счётчик
weigth = 2

while True:
    ret, frame = cap.read()
    if ret == True:
        img = cv2.resize(frame, (72, 128))
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#удаляем лиишние цветовые каналы

        if i == 3825:
            weigth = 3
        if i == 6000:
            weigth = 2
        if i == 13750:
            weigth = 1

        if i % 10 == 0:
            x = torch.from_numpy(img_gray)
            x = x.float()
            x = x.reshape([72 * 128])
            x_validation.append(x)
            y_validation.append(weigth)
        i += 1
    elif i > 19500:
        break
cap.release()#освобождаем камеру

#Запись вадидационной выборки в файл


f = open(r'x_validation.txt', 'wb')
pickle.dump(x_validation, f)
f.close()


f = open(r'y_validation.txt', 'wb')
pickle.dump(y_validation, f)
f.close()
