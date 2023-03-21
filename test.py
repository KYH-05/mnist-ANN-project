#-----------------------------------------------------------------------------------------------------------------------
#MNIST project
#학습 검증
#-------------------------------------------------------------------------
#module
from data import load_mnist
import random
import numpy as np
from tqdm import tqdm
import pickle
import os
#-----------------------------------------------------------------------------
#(훈련 이미지, 훈련 레이블), (시험 이미지, 시험 레이블)
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
#-----------------------------------------------------------------------------
#계산함수
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
#변수설정1
size1=784
size2=16
size3=16
size4=10
a=int(input("epoch:"))
b=int(input(" iteration:"))
c=int(input("batch_size:"))
with open('weight.pkl'+'$'+str(a)+'$'+str(b)+'$'+str(c), 'rb') as f:
  w1 = pickle.load(f)
  w2 = pickle.load(f)
  w3 = pickle.load(f)
true=0#답이 맞았는지
cost=0
#------------------------------------------------------------------------------
#행렬연산
def calculation(i):
  global n1,n2_1,n2_2,n3_1,n3_2,n4_1,n4_2
  n1=x_test[i]
  n2_1 = np.dot(n1, w1)
  n2_2 = ReLU(n2_1)
  n3_1 = np.dot(n2_2, w2)
  n3_2 = ReLU(n3_1)
  n4_1 = np.dot(n3_2, w3)
  n4_2=softmax(n4_1)
  n4_2 = n4_2.tolist()
#------------------------------------------------------------------------------
def check(i):
  global cost,true
  true_value=[0 for p in range(0,size4)]
  true_value[t_test[i]]=1
  for p in range(0,size4):
    cost+=-true_value[p]*np.log(n4_2[p]+(1/10)**8)
  if n4_2.index(max(n4_2))==t_test[i]:
    true+=1
  if i==9999:
    print("accuracy:",(true/(10000))*100,"%")
    print("average loss",cost/10000)
#------------------------------------------------------------------------------
#실행
print("testing...")
for i in tqdm(range(0,10000)):
  calculation(i)
  check(i)
print(a,b,c)
os.system("pause")

