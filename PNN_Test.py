import math
import numpy as np
import pandas as pd

file = pd.read_csv('data_train.txt', delimiter = "\t")
file_test = pd.read_csv('data_test.txt', delimiter="\t")

class0 = file.loc[file['label'] == 0]
class1 = file.loc[file['label'] == 1]
class2 = file.loc[file['label'] == 2]

## MENCARI NILAI SIGMA##
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

def find_distance_total(C):
        return np.sum(find_distance(i) for i in index_for_category(C))

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
#################ENDING#####################################


#Membuat Arsitektur PNN

#Hidden Layer
sigma = 5.48891736
tempLabel = []
summation0 = 0
summation1 = 0
summation2 = 0
cek = 0
for i in range(len(file_test)):
    for j in range(len(file)):
            hidden = np.exp([-(((file_test.loc[i, 'att1'] - file.loc[j, 'att1']) ** 2) +
                               (file_test.loc[i, 'att2'] - file.loc[j, 'att2']) ** 2 +
                               (file_test.loc[i, 'att3'] - file.loc[j, 'att3']) ** 2) / 2 * sigma ** 2])
            if (file.loc[j, 'label'] == 0):
                summation0 = np.sum(hidden)
            elif(file.loc[j, 'label'] == 1):
                summation1 = np.sum(hidden)
            elif(file.loc[j, 'label'] == 2):
                summation2 = np.sum(hidden)

    if (max(summation0, summation1, summation2) == summation0):
        tempLabel.append(0)
    elif (max(summation0, summation1, summation2) == summation1):
        tempLabel.append(1)
    elif (max(summation0, summation1, summation2) == summation2):
        tempLabel.append(2)

print(tempLabel)
prediksi = open("File_Predisksi.txt","w")
j = ''
for i in tempLabel:
    j = j + str(i) + "\n"
prediksi.write(j)




















#Membuat Summation Layer
