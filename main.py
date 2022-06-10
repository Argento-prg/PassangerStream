from PyQt5 import uic
from PyQt5 import QtWidgets

import sys

import numpy as np
import torch
import torch.nn.functional as F
import random
import cv2





#для одинаковой генерации случайных чисел
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

#Объявляем класс нейросети, чтобы не было проблем с импортом обученной модели
class VISIONNet(torch.nn.Module):
    def __init__(self, n_hidden_neurons):
        super(VISIONNet, self).__init__()
        self.fc1 = torch.nn.Linear(72 * 128, n_hidden_neurons)
        self.fc2 = torch.nn.Linear(n_hidden_neurons, n_hidden_neurons)
        self.fc3 = torch.nn.Linear(n_hidden_neurons, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x)

#загружаем шаблон формы
Form, _ = uic.loadUiType("PassangerStream_form.ui")
#загружаем обученную модель нейросети
model = torch.load("mymodel.pb")

class Ui(QtWidgets.QMainWindow,Form):

    path_video = ""#текущий путь
    flag_work = False#флаг того, что сейчас запущена работа
    flag_path = False#флаг того, что путь есть
    current_status = -1#текущий статус выхода нейросети
    pred_path_video = ""#предыдущий путь
    count = 0
    sum = 0
    flag_stop = False


    def __init__(self):
        super(Ui, self).__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.act_work)
        openFile = QtWidgets.QAction('Open', self)
        openFile.setShortcut('Ctrl+O')
        openFile.setStatusTip('Open new File')
        openFile.triggered.connect(self.choose_video)
        self.menuopfile.addAction(openFile)
        self.pushButton_stop.clicked.connect(self.stop)

    #действие при нажатии на кнопку Stop
    def stop(self):
        self.flag_stop = True
        if not self.flag_path:
            self.flag_path = True



    #Действие при нажатии на кнопку pushButton(Старт)
    def act_work(self):
        if self.flag_work and self.flag_path:
            self.flag_path = False
            self.flag_stop = False
            self.flag_work = False
            cap = cv2.VideoCapture(self.path_video)#загружаем выбранное видео

            temp = []
            while True:
                #если user выбрал другое видео, но не нажал кнопку, мы завершаем цикл
                if self.path_video != self.pred_path_video:
                    if self.pred_path_video != "":
                        self.pred_path_video = self.path_video
                        break

                ret, frame = cap.read()
                if ret == True:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    cv2.imshow('frame', frame)
                    cv2.waitKey(1)
                    self.current_status = self.process_frame(frame)
                    temp.append(self.current_status)
                    self.count += 1
                    if self.count == 25:#среднее арифметическое из 25 кадров
                        self.count = 24
                        for i in temp:
                            self.sum += int(i)
                        result = self.sum / 25
                        self.sum = 0
                        temp.pop(0)
                        self.print_answer(result)
                else:
                    self.sum = 0
                    temp.clear()
                    self.count = 0
                    break

                if self.flag_stop:
                    self.sum = 0
                    temp.clear()
                    self.count = 0
                    break

            cap.release()#освобождаем камеру
            cv2.destroyAllWindows()#разрушаем окна opencv
            self.flag_work = True
            self.print_answer(0)#Обнуляем оценку


    #Выбор видео
    def choose_video(self):
        self.path_video = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', '~')[0]
        self.flag_path = True
        self.flag_work = True


    #Обработка кадра
    def process_frame(self, frame):
        img = cv2.resize(frame, (72, 128))#сжимаем изображение в 100 раз
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#убираем лишние цветовые каналы
        x = torch.from_numpy(img_gray)#преобразуем массив numpy в tensor
        x = x.float()
        x = x.reshape([72 * 128])#изменяем размерность тензора
        prediction = model.forward(x)#делаем предсказание
        answer = prediction.argmax()#считываем выход
        return answer

    #Вывод ответа нейросети
    def print_answer(self, ans):
        self.progressBar.setValue(int(ans * 100/ 2))#перевод ответа в процентную оценку

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = Ui()
    w.show()
    sys.exit(app.exec_())
