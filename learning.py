#-----------------------------------------------------------------------------------------------------------------------
#MNIST project
#ANN
#hard coding
#bp
#mini_batch,adam,notuse_HE
#-------------------------------------------------------------------------
#module
from data import load_mnist
import random
import numpy as np
from tqdm import tqdm
import pickle
import os
#-----------------------------------------------------------------------------
#데이터 불러오기->(훈련 이미지, 훈련 답), (검증 이미지, 검증 답)
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False) #1차원 배열로, 0~1로 정규화
#-----------------------------------------------------------------------------
#변수설정1->신경망 구성
size1=784 #layer_size
size2=16
size3=16
size4=10
weight_num=(size1*size2)+(size2*size3)+(size3*size4)
print(weight_num)
w1 = [[((random.random()*2)-1)/100 for p in range(0, size2)] for q in range(0, size1)] #weight,HE사용x
w2 = [[((random.random()*2)-1)/100 for p in range(0, size3)] for q in range(0, size2)]
w3 = [[((random.random()*2)-1)/100 for p in range(0, size4)] for q in range(0, size3)]
#------------------------------------------------------------------------------
#변수설정2->gradient로 weight_change 구하기
batch_size=16
ws=(sum(w3, []))+(sum(w2, []))+(sum(w1, [])) #가장 끝 위에서부터 w1/왼쪽고정하고 오른쪽 내림
gradients=[0 for p in range(0,weight_num)]#기울기/미니배치에서 1개 할때 기울기 모두 구하고 adam (-learningrate*(v/(s)**(1/2)+10**-10))
print(len(gradients))
weight_changes=[0 for p in range(0,weight_num)]#adam 게산
mini_batch_weight_changes=[[0 for p in range(0,batch_size)]for q in range(0,weight_num)]#가로는 미니배치,세로는 가중치
average_weight_change=[0 for p in range(0,weight_num)]#평균내서 ws에 업데이트하고 w1,w2,w3로 변경
#------------------------------------------------------------------------------
#변수설정3->adam
v = [0 for p in range(0, weight_num)]
s = [0 for p in range(0, weight_num)]
b1 = 0.9
b2 = 0.999
rate = 0.001
#------------------------------------------------------------------------------
#계산함수
def ReLU(x):
  return np.maximum(0, x)
def ReLU_d(x):
  if ReLU(x)>0:
    D=1
  else:
    D=0
  return D
def sigmoid(x):
  return 1 / (1 + np.exp(-x))
#------------------------------------------------------------------------------
#행렬연산
def calculation(i):
  global n1,n2_1,n2_2,n3_1,n3_2,n4_1,n4_2
  n1=x_train[i]
  n2_1 = np.dot(n1, w1)
  n2_2 = ReLU(n2_1)
  n3_1 = np.dot(n2_2, w2)
  n3_2 = ReLU(n3_1)
  n4_1 = np.dot(n3_2, w3)
  n4_2=sigmoid(n4_1)
  n4_2 = n4_2.tolist()
#------------------------------------------------------------------------------
#t_train으로 기대되는 값 리스트로 표현
def value(i):
  global true_value
  true_value=[0 for a in range(len(n4_2))]
  true_value[t_train[i]]=1
  #cost=0
  #for p in range(0,len(n4_2)):
    #cost+=(true_value[p]-n4_2[p])**(2)
  #print(i,cost)
#------------------------------------------------------------------------------
#역전파(미니배치)
def backpropagation(i):
  global gradients
  list1=[0 for p in range(0,size4)]
  list2=[0 for p in range(0,size3)]
  list3=[0 for p in range(0,size2)]
  #----
  for p in range(0,size4):#list1
    list1[p]=(1/size4)*2*(n4_2[p]-true_value[p]) * (sigmoid(n4_1[p])*(1-sigmoid(n4_1[p])))
  A_m = 0  # n3-n4s다발들->list2
  for p in range(0, size3):
    for q in range(0, size4):
      A_m += list1[q] * w3[p][q]
    list2[p] = A_m
    A_m = 0
  B_m = 0  # n2-n3s-n4s다발들->list3
  for p in range(0, size2):
    for q in range(0, size3):
      B_m += (list2[q]) * (ReLU_d(n3_1[q])) * (w2[p][q])
    list3[p] = B_m
    B_m = 0
  #----
  for p in range(0,size3):#w3
    for q in range(0,size4):
      gradients[p*size4+q]=list1[q]*n3_2[p]
  for p in range(0,size2):#w2
    for q in range(0,size3):
      gradients[size3 * size4 + (p * size3 + q)]= (list2[q]) * (ReLU_d(n3_1[q])) *  n2_2[p]
  for p in range(0,size1):#w1
    for q in range(0,size2):
      gradients[size3*size4+size2*size3+(p*size2+q)]=(list3[q]) * (ReLU_d(n2_1[q])) * (w1[p][q])
#------------------------------------------------------------------------------
def weight_update(i):#gradient->adam->weight_changes->mini_batch_weight_changes->리스트 꽉 차면->average->ws->w123
  global w1,w2,w3
  for p in range(0,weight_num):
    v[p] = b1 * v[p] + (1 - b1) * gradients[p]
    s[p] = b2 * s[p] + (1 - b2) * ((gradients[p]) ** 2)
    v_t=v[p]/(1-(b1)**(a+1))
    s_t=s[p]/(1-(b2)**(a+1))
    Q = v_t /(((s_t)**(1/2))+(1/(10**8)))
    weight_changes[p]= -rate * Q
    mini_batch_weight_changes[p][b] = weight_changes[p]  #b는 전체 실행 부분의 batch/가로16세로12960에서 세로로 넣기
  #----
  if b==batch_size-1:#mini_batch_weight_changes에 값이 모두 들어가면
    for p in range(0,12960):
      average_weight_change[p]=sum(mini_batch_weight_changes[p])/(batch_size)#평균내기
      ws[p]=ws[p]+average_weight_change[p]#업데이트
    w3=np.reshape(ws[0:size3*size4],(size3,size4))#ws 형태변환
    w2=np.reshape(ws[size3*size4:size3*size4+size2*size3],(size2,size3))
    w1=np.reshape(ws[size3*size4+size2*size3:size3*size4+size2*size3+size1*size2],(size1,size2))
    w3 = w3.tolist()
    w2 = w2.tolist()
    w1 = w1.tolist()
#------------------------------------------------------------------------------
#실행
epoch=int(60000/batch_size)
batch_size=int(batch_size)
print("learning...")
for a in tqdm(range(0,epoch)):
  for b in range(0,batch_size):
    calculation(a*batch_size+b)
    value(a*batch_size+b)
    backpropagation(a*batch_size+b)
    weight_update(a*batch_size+b)
#------------------------------------------------------------------------------
# 파일로 저장
with open('weight.pkl', 'wb') as f:
    pickle.dump(w1, f)
    pickle.dump(w2, f)
    pickle.dump(w3, f)
os.system("pause")
