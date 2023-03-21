# Multi_classfication

data: MNIST

networks: ANN

learn: Backpropagation

batch: Mini_batch

optimizer: Adam

weight_initialization: He

activation_function: Relu,softmax

cost_funtion: Crossentropy,k/Dispersion

# module 
tqdm,pickle,keyboard,PIL

# help
train: mnist data로 ANN 학습, 대략 2 epoch(5시간)에서 98%의 accuracy를 지님

test: 10000개의 mnist데이터로 학습 수준 검증

exam: 직접 손글씨 이미지(png)를 만든 후 ANN으로 테스트
