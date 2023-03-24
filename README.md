# Number Handwriting Multi Classfication

data: MNIST

networks: ANN

learn: Backpropagation

batch: Mini_batch

optimizer: Adam

weight_initialization: He

activation_function: Relu,softmax

cost_funtion: Crossentropy,k/Dispersion

# module 
tqdm, pickle, keyboard, PIL

# help
data: mnist data

train: mnist data로 ANN 학습/ 대략 2 epoch(5시간)에서 98%의 accuracy를 지님

test: 10000개의 mnist데이터로 학습 수준 검증

image: 직접 손글씨 이미지(png)를 만든 후 ANN으로 테스트

retrain: train.py로 학습된 가중치, 편향 파일인 weight.pkl을 불러와 다시 학습 재개

weight.pkl$2$3750$7: train.py을 통해 학습된 weight,bias/ test.py,exam.py에서 사용가능


