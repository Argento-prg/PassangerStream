import matplotlib.pyplot as plt
import numpy as np
import torch
import pickle
import torch.nn.functional as F
import random

#обнуляем рандомные значения
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

#инициализация данных
f = open(r'x_train.txt', 'rb')
x_train = pickle.load(f)
f.close()

f = open(r'y_train.txt', 'rb')
y_train = pickle.load(f)
f.close()

f = open(r'x_validation.txt', 'rb')
x_validation = pickle.load(f)
f.close()

f = open(r'y_validation.txt', 'rb')
y_validation = pickle.load(f)
f.close()

x_train = torch.stack(x_train)
y_train = torch.tensor(y_train) - 1
x_validation = torch.stack(x_validation)
y_validation = torch.tensor(y_validation) - 1



#Класс нейросети
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
        #ReLu - усечённое линейное преобразование


vision_net = VISIONNet(100)

loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(vision_net.parameters(), lr=0.01)

batch_size = 100

#ОБУЧЕНИЕ НЕЙРОСЕТИ
print("Начало обучения:\n")#лог
for epoch in range(2000):
    order = np.random.permutation(len(x_train))#перемешивание выборки

    for start_index in range(0, len(x_train), batch_size):
        optimizer.zero_grad()#обнуляем градиент

        batch_indexes = order[start_index:start_index + batch_size]

        x_batch = x_train[batch_indexes]
        y_batch = y_train[batch_indexes]

        preds = vision_net.forward(x_batch)#предсказания на тренировочной выборке
        loss_value = loss(preds, y_batch)#функция потерь
        loss_value.backward()#градиент от результата функции потерь

        optimizer.step()#оптимизируем веса нейронов

    test_preds = vision_net.forward(x_validation)#валидация


    accuracy = (test_preds.argmax(dim=1) == y_validation).float().mean()
    print("Эпоха № " + str(epoch) + " " + str(accuracy))
    if accuracy > 0.8:#прерываем цикл, если точность предсказаний больше 80%
        break


print("Обучение закончено!")#лог

print("\nСохранить?\n")
if input() == "yes":
    torch.save(vision_net, "mymodel.pb")#сохраняем обученную модель
