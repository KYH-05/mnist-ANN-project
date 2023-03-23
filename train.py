#-----------------------------------------------------------------------------------------------------------------------
#data: MNIST
#problem: Multi_classification
#networks: ANN
#learn: Backpropagation
#batch: Mini_batch
#optimizer: Adam
#weight_initialization: He
#activation_function: Relu,softmax
#cost_funtion: Crossentropy,k/Dispersion
#-------------------------------------------------------------------------
#module
from data import load_mnist
import random
import numpy as np
from tqdm import tqdm
import pickle
import os
import time
import keyboard
#-----------------------------------------------------------------------------
#data: (훈련 이미지,훈련 답), (테스트 이미지,테스트 답)
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False) #1차원 배열로, 0~1로 정규화
train_data_size=16*3750
test_data_size=10000
#-----------------------------------------------------------------------------
#변수설정1:신경망 구성
size1=784 #layer_size
size2=16
size3=16
size4=10
weight_num=(size1*size2)+(size2*size3)+(size3*size4)
bias_num=size2+size3+size4
w1 = [[(np.random.randn()/np.sqrt(size1/2)) for p in range(0, size2)] for q in range(0, size1)] #weight(left->right/우상시작,좌축)
w2 = [[(np.random.randn()/np.sqrt(size2/2)) for p in range(0, size3)] for q in range(0, size2)]
w3 = [[(np.random.randn()/np.sqrt(size3/2)) for p in range(0, size4)] for q in range(0, size3)]
b1=[0.01 for p in range(0,size2)] #bias(left->right)
b2=[0.01 for p in range(0,size3)]
b3=[0.01 for p in range(0,size4)]
wb_s=(sum(w3, []))+(sum(w2, []))+(sum(w1, []))+b3+b2+b1 #w3->w1으로 역전파
#------------------------------------------------------------------------------
#변수설정2:adam
v = [0 for p in range(0, weight_num+bias_num)]
s = [0 for p in range(0, weight_num+bias_num)]
k1 = 0.9
k2 = 0.999
rate = 0.01
#------------------------------------------------------------------------------
#변수설정3: gradient->weight_change
batch_size=16
gradients=[0 for p in range(0,weight_num+bias_num)]#기울기
changes=[0 for p in range(0,weight_num+bias_num)]#adam 게산(change=-rate*Q)
mini_batch_changes=[[0 for p in range(0,batch_size)]for q in range(0,weight_num+bias_num)]#가로는 미니배치,세로는 가중치와 편향
average_change=[0 for p in range(0,weight_num+bias_num)]#mini_batch_changes에서 가로로 평균
#------------------------------------------------------------------------------
#활성화함수
def ReLU(x):
  return np.maximum(0, x)
def ReLU_d(x):
  if ReLU(x)>0:
    D=1
  else:
    D=0
  return D
def softmax(x):
  c = np.max(x)
  exp_x = np.exp(x - c)
  sum_exp_x = np.sum(exp_x)
  y = exp_x / sum_exp_x
  return y
#------------------------------------------------------------------------------
#epoch마다 변수초기화
def initialization(i):
  global cost,true,x_train,t_train,cost2
  cost=0
  cost2=0
  true=0
  idx = np.arange(x_train.shape[0])
  np.random.shuffle(idx)
  x_train = x_train[idx]
  t_train = t_train[idx]
#------------------------------------------------------------------------------
#행렬연산
def calculation(i):
  global n1,n2_1,n2_2,n3_1,n3_2,n4_1,n4_2,b1,b2,b3
  n1=x_train[i]
  n2_1 = np.dot(n1, w1)+b1
  n2_2 = ReLU(n2_1)
  n3_1 = np.dot(n2_2, w2)+b2
  n3_2 = ReLU(n3_1)
  n4_1 = np.dot(n3_2, w3)+b3
  n4_2=softmax(n4_1)
  n4_2 = n4_2.tolist()
#-----------------------------------------------------------------------------
#역전파
def backpropagation(i):
  global gradients,true_value,bias,T
  true_value = [0 for a in range(len(n4_2))]
  true_value[t_train[i]] = 1
  delta=(1/10)**8
  T=0.1
  list1=[(n4_2[p]-true_value[p])-T*(2*n4_2[p]-0.2)/((n4_2[p]**2)-0.2*n4_2[p]+delta) * (n4_2[p]*(1- n4_2[p])) for p in range(0,size4)]
  list2=[sum([(list1[q] * w3[p][q]) for q in range(0, size4)]) for p in range(0, size3)]
  list3=[sum([((list2[q]) * (ReLU_d(n3_1[q])) * (w2[p][q])) for q in range(0, size3)]) for p in range(0, size2)]
  weight_g=[(list1[q]*n3_2[p]) for p in range(0,size3) for q in range(0,size4)]\
            +[(list2[q]) * (ReLU_d(n3_1[q])) * n2_2[p] for p in range(0,size2) for q in range(0,size3)]\
            +[(list3[q]) * (ReLU_d(n2_1[q])) * (n1[p]) for p in range(0,size1) for q in range(0,size2)]
  bias_g=[(list1[p]) for p in range(0,size4)]+[(list2[p]) * (ReLU_d(n3_1[p])) for p in range(0,size3)]+[(list3[p]) * (ReLU_d(n2_1[p])) for p in range(0,size2)]
  gradients=weight_g+bias_g
#------------------------------------------------------------------------------
#가중치 변경(gradient->adam->changes->mini_batch_hanges->리스트 꽉 차면->average_change->wb_s->ws,bs->w1,w2,w3,b1,b2,b3)
def weight_update(i):
  global w1,w2,w3,b1,b2,b3,ws,bs,wb_s,gradients
  for p in range(0, weight_num+bias_num):
    v[p] = k1 * v[p] + (1 - k1) * gradients[p]
    s[p] = k2 * s[p] + (1 - k2) * ((gradients[p]) ** 2)
    v_t = v[p] / (1 - (k1) ** (c * iteration + a + 1))
    s_t = s[p] / (1 - (k2) ** (c * iteration + a + 1))
    Q = v_t / (((s_t) ** (1 / 2)) + (1 / (10 ** 8)))
    changes[p] = -rate * Q
    mini_batch_changes[p][b] = changes[p]  # b는 전체 실행 부분의 batch/가로16 세로num에서 세로로 넣기
  if b==batch_size-1:#mini_batch_weight_changes에 값이 모두 들어가면
    for p in range(0,weight_num+bias_num):
      average_change[p]=sum(mini_batch_changes[p])/(batch_size)#평균내기
      wb_s[p]=wb_s[p]+average_change[p]#업데이트
    ws=wb_s[0:weight_num]#w1,w2,w3,b1,b2,b3 형태변환
    bs=wb_s[weight_num:]
    w3=np.reshape(ws[0:size3*size4],(size3,size4))
    w2=np.reshape(ws[size3*size4:size3*size4+size2*size3],(size2,size3))
    w1=np.reshape(ws[size3*size4+size2*size3:size3*size4+size2*size3+size1*size2],(size1,size2))
    b3=bs[0:size4]
    b2 = bs[size4:size4+size3]
    b1 = bs[size4+size3:]
    w3 = w3.tolist()
    w2 = w2.tolist()
    w1 = w1.tolist()
#------------------------------------------------------------------------------
#cost,accuracy 계산
def recording(i):
  global cost,true,true_value,accuracy,cost2,T
  for p in range(0, len(n4_2)):
    cost += -true_value[p] * np.log(n4_2[p] + (1 / 10) ** (8))
    cost2+=T*(-0.9+(1/((n4_2[0]-0.1)**2+(n4_2[1]-0.1)**2+(n4_2[2]-0.1)**2+(n4_2[3]-0.1)**2+(n4_2[4]-0.1)**2+(n4_2[5]-0.1)**2+(n4_2[6]-0.1)**2+(n4_2[7]-0.1)**2+(n4_2[8]-0.1)**2+(n4_2[9]-0.1)**2)))
  if n4_2.index(max(n4_2))==t_train[i]:
    true+=1
  if i==train_data_size-1:
    cost=cost/train_data_size
    cost2=cost2/train_data_size
  if i==train_data_size-1:
    accuracy=(true/train_data_size)*100
#------------------------------------------------------------------------------
#실행
epoch=10000
iteration=int(train_data_size/batch_size)
batch_size=int(batch_size)
print("train.py")
print("weight,bias output->press p")
for c in range(0,epoch):
  print("Epoch:",(c+1))
  initialization(c)
  for a in tqdm(range(0,iteration)):
    for b in range(0,batch_size):
      calculation(a*batch_size+b)
      backpropagation(a*batch_size+b)
      weight_update(a*batch_size+b)
      recording(a * batch_size + b)
      if keyboard.is_pressed("p"):
        with open('weight.pkl'+'$'+str(c+1)+'$'+str(a+1)+'$'+str(b+1), 'wb') as f:
          pickle.dump(w1, f)
          pickle.dump(w2, f)
          pickle.dump(w3, f)
          pickle.dump(b1, f)
          pickle.dump(b2, f)
          pickle.dump(b3, f)
  print("Accuracy:",accuracy,"Cost1",cost,"Cos2t",cost2,"Cost_sum",cost+cost2)
  print()
#------------------------------------------------------------------------------
# 파일로 저장
with open('weight.pkl'+'$'+str(epoch)+'$'+str(iteration)+'$'+str(batch_size), 'wb') as f:
    pickle.dump(w1, f)
    pickle.dump(w2, f)
    pickle.dump(w3, f)
    pickle.dump(b1, f)
    pickle.dump(b2, f)
    pickle.dump(b3, f)
os.system("pause")

