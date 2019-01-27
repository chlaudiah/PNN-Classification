import math
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import csv


file = pd.read_csv('data_train.txt', delimiter = "\t")

class0 = file.loc[file['label'] == 0]
class1 = file.loc[file['label'] == 1]
class2 = file.loc[file['label'] == 2]

#Tahap Pertama
W = file[['att1', 'att2', 'att3']].values
print('W = %s' % W)
print()

def index_for_category(C_indeks):
    match_index = list(file[file.label == C_indeks].index)
    print('Data untuk Class= %s: %s' % (C_indeks, match_index))
    return match_index
print()

Class= file.label.unique()
print('Classes: %s' % Class)

index_for_category(Class[0])
index_for_category(Class[1])
index_for_category(Class[2])

def count_label(C_indeks):
    return len(file[file.label == C_indeks])

print()

for C_indeks in Class:
    Total_data = [count_label(C_indeks)]
    print('Total Data: %s' % Total_data)
print()

#Tahap Kedua
def find_distance(i):
    C_indeks = file.label[i]
    indeks = np.where(Class == C_indeks)[0][0]
    indexes = index_for_category(C_indeks)
    indexes.remove(i)
    print(i, C_indeks, indeks, indexes)
    print()
    distance_list = [np.linalg.norm(W[i] - W[index]) for index in indexes]
    distance = np.amin(distance_list) or 1.0
    print("Distance list: %s -> Distance: %s " % (distance_list, distance))
    return distance

print(find_distance(1))

def find_distance_total(C):
        return np.sum(find_distance(i) for i in index_for_category(C))

print()

distance_total = np.array([find_distance_total(i) for i in Class])
print("Distance Total[0]= %s" % distance_total[0])
print("Distance Total[1]= %s" % distance_total[1])
print("Distance Total[2]= %s" % distance_total[2])

g = np.random.uniform(1,10)
print()
print("Nilai g: ",g)

distance_rata2 = distance_total / Total_data
print()
print("Jarak rata-rata: %s" % distance_rata2)

sigma = g * distance_rata2
print("Nilai Sigma: %s" % sigma)



#Membuat Hidden Layer

print()
for i in range(0,99):
    train = np.array([(file['att1'][i], file['att2'][i], file['att3'][i], file['label'][i])])

print()
for i in range(100, 150):
    validasi = np.array([(file.loc[i,'label'])])
    # print(train)

# print()

#Hidden Layer
sigma = 5.48891736
tempLabel = []
summation0 = 0
summation1 = 0
summation2 = 0

for i in range(100, 150):#validasi
    for j in range(0,99):#train
            hidden = np.exp([-(((file.loc[i, 'att1'] - file.loc[j, 'att1']) ** 2) +
                               (file.loc[i, 'att2'] - file.loc[j, 'att2']) ** 2 +
                               (file.loc[i, 'att3'] - file.loc[j, 'att3']) ** 2) / 2 * sigma ** 2])
            if (file.loc[j, 'label'] == 0):
                summation0 = np.sum(hidden)
            elif(file.loc[j, 'label'] == 1):
                summation1 = np.sum(hidden)
            elif(file.loc[j, 'label'] == 2):
                summation2 = np.sum(hidden)
    #Membuat Summation Layer
    if (max(summation0, summation1, summation2) == summation0):
        tempLabel.append(0)
    elif (max(summation0, summation1, summation2) == summation1):
        tempLabel.append(1)
    elif (max(summation0, summation1, summation2) == summation2):
        tempLabel.append(2)

index_train = 100
false = 0
for i in tempLabel:
    if(file.loc[index_train, 'label'] != i):
        false = false+1
    print(index_train, file.loc[index_train, 'label'], i)
    index_train = index_train+1
print("Hasil Akurasi: ", ((50-false)/50)*100)
