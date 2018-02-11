from keras.layers import Input, Dense
from keras.models import Model

from IntegratedGradients import *

X_train = np.array([[float(j) for j in i.rstrip().split(",")]
                    for i in open("train.csv").readlines()])
Y_train = X_train[:,-1]
X_train = X_train[:,0:-1]

X_test = np.array([[float(j) for j in i.rstrip().split(",")]
                    for i in open("test.csv").readlines()])
Y_test = X_test[:,-1]
X_test = X_test[:,0:-1]

inputs = Input(shape=[9])

x = Dense(64, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)

probs = Dense(1, activation='sigmoid')(x)


model1 = Model(inputs=inputs, outputs=probs)
model1.compile(optimizer='sgd', loss='binary_crossentropy')

model1.fit(X_train, Y_train, epochs=1000, batch_size=128,
           validation_split=0.15, verbose=True)

int_gradients = integrated_gradients(model1)

def print_results(title, x):

  def to_ints(v):
    v /= np.max(np.abs(v), axis=0)
    return [int(100.0 * x) for x in v]

  int_to_name = ['Clump Thickness','Uniformity of Cell Size','Uniformity of Cell Shape',
                 'Marginal Adhesion','Single Epithelial Cell Size','Bare Nuclei',
                 'Bland Chromatin','Normal Nucleoli','Mitoses']

  ints = to_ints(x)
  print("** Contributions to classification for sample type ", title, " **")
  for i in range(9):
    print("\t", int_to_name[i],":\t", ints[i])

print_results("benign sample", int_gradients.explain(X_test[8], num_steps=200))
print_results("malignant sample", int_gradients.explain(X_test[0], num_steps=200))

