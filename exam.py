from PIL import Image
import numpy as np
import os
import pickle
#------------------------------------------------------------------------------
print("<이미지 선택>")
num=int(input("number:"))
print()
img = Image.open(str(num)+".png").convert("L")
img = np.resize(img, (1, 784))
test_data = ((np.array(img) / 255) - 1) * -1
#------------------------------------------------------------------------------
print("<가중치 선택>")
a=int(input("epoch:"))
b=int(input(" iteration:"))
c=int(input("batch_size:"))
with open('weight.pkl'+'$'+str(a)+'$'+str(b)+'$'+str(c), 'rb') as f:
  w1 = pickle.load(f)
  w2 = pickle.load(f)
  w3 = pickle.load(f)
#------------------------------------------------------------------------------
def ReLU(x):
  return np.maximum(0, x)
def sigmoid(x):
  return 1 / (1 + np.exp(-x))
def softmax(x):
  c = np.max(x)
  exp_x = np.exp(x - c)
  sum_exp_x = np.sum(exp_x)
  y = exp_x / sum_exp_x
  return y
#------------------------------------------------------------------------------
n1=test_data
n2_1 = np.dot(n1, w1)
n2_2 = ReLU(n2_1)
n3_1 = np.dot(n2_2, w2)
n3_2 = ReLU(n3_1)
n4_1 = np.dot(n3_2, w3)
n4_2=softmax(n4_1)
n4_2 = n4_2.tolist()
#------------------------------------------------------------------------------
print("<결과>")
print(n4_2.index(max(n4_2)))
os.system("pause")