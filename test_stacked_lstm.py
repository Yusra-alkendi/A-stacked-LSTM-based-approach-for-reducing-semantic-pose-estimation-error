import tensorflow as tf 
from tensorflow import keras 
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import itertools
import numpy 
import random
from itertools import product
import math
import csv
from numpy.random import seed
from sklearn.model_selection import train_test_split
from sklearn.externals.joblib import dump, load
import os
import sys 

def write_file (data, file_name):

  dir = os.path.join('dataset', 'dataset_random_train_validate_test')
  if not os.path.exists(dir):
    os.mkdir(dir)
  with open(dir+'/'+file_name, 'w') as csvFile:
      writer = csv.writer(csvFile)
      writer.writerows(data)
  csvFile.close()


def convert_trajectory_to_sequence (seq_length, estimate, ground_truth) :

        
    sequences=numpy.zeros((seq_length*(estimate.shape[0]-seq_length), 3))
    target_poses=numpy.zeros((estimate.shape[0]-seq_length,3))
    input_poses=numpy.zeros((estimate.shape[0]-seq_length,3))
    for i in range (0,ground_truth.shape[0]-seq_length):
        index=i+seq_length
        sequences[i*seq_length:(i+1)*seq_length,:]=estimate[i:index, :]
        target_poses[i,:]=ground_truth[i+seq_length-1, :]
        input_poses[i,:]=estimate[i+seq_length-1, :]
        
    sequences = numpy.reshape(sequences, (int(sequences.shape[0]/seq_length), seq_length,3))

    return sequences, target_poses, input_poses 

class Swish(tf.keras.layers.Activation):
    
    def __init__(self, activation, **kwargs):
        super(Swish, self).__init__(activation, **kwargs)
        self.__name__ = 'swish'

def swish(x):
    A = (tf.keras.backend.sigmoid(x) * x)
    return A

tf.keras.utils.get_custom_objects().update({'swish': Swish(swish)})

#layer_size=int(sys.argv[1])
layer_size=256
num_layers=2
seq_length=30
estimate_exp1=numpy.loadtxt("dataset/exp1/estimate.txt", delimiter=" ")
estimate_exp2=numpy.loadtxt("dataset/exp2/estimate.txt", delimiter=" ")
estimate_exp3=numpy.loadtxt("dataset/exp3/estimate.txt", delimiter=" ")
estimate_exp4=numpy.loadtxt("dataset/exp4/estimate.txt", delimiter=" ")
estimate_exp5=numpy.loadtxt("dataset/exp5/estimate.txt", delimiter=" ")
estimate_exp6=numpy.loadtxt("dataset/exp6/estimate.txt", delimiter=" ")
estimate_exp7=numpy.loadtxt("dataset/exp7/estimate.txt", delimiter=" ")
estimate_exp8=numpy.loadtxt("dataset/exp8/estimate.txt", delimiter=" ")
estimate_exp9=numpy.loadtxt("dataset/exp9/estimate.txt", delimiter=" ")
estimate_exp10=numpy.loadtxt("dataset/exp10/estimate.txt", delimiter=" ")
estimate_exp11=numpy.loadtxt("dataset/exp11/estimate.txt", delimiter=" ")
estimate_exp12=numpy.loadtxt("dataset/exp12/estimate.txt", delimiter=" ")
estimate_exp13=numpy.loadtxt("dataset/exp13/estimate.txt", delimiter=" ")
estimate_exp14=numpy.loadtxt("dataset/exp14/estimate.txt", delimiter=" ")
estimate_exp15=numpy.loadtxt("dataset/exp15/estimate.txt", delimiter=" ")
estimate_exp16=numpy.loadtxt("dataset/exp16/estimate.txt", delimiter=" ")
estimate_exp17=numpy.loadtxt("dataset/exp17/estimate.csv", delimiter=";")
estimate_exp18=numpy.loadtxt("dataset/exp18/estimate.csv", delimiter=";")
estimate_exp19=numpy.loadtxt("dataset/exp19/estimate.csv", delimiter=";")


estimate_exp1=estimate_exp1[0:2200,1:4]
estimate_exp2=estimate_exp2[0:1300,1:4]
estimate_exp3=estimate_exp3[0:1100,1:4]
estimate_exp4=estimate_exp4[0:1100,1:4]
estimate_exp5=estimate_exp5[0:1200,1:4]
estimate_exp6=estimate_exp6[0:1700,1:4]
estimate_exp7=estimate_exp7[0:2100,1:4]
estimate_exp8=estimate_exp8[0:2100,1:4]
estimate_exp9=estimate_exp9[0:4000,1:4]
estimate_exp10=estimate_exp10[0:2700,1:4]
estimate_exp11=estimate_exp11[0:1700,1:4]
estimate_exp12=estimate_exp12[0:700,1:4]
estimate_exp13=estimate_exp13[0:1000,1:4]
estimate_exp14=estimate_exp14[0:1500,1:4]
estimate_exp15=estimate_exp15[0:2300,1:4]
estimate_exp16=estimate_exp16[0:1200,1:4]
estimate_exp17=estimate_exp17[0:420,0:3]
estimate_exp18=estimate_exp18[0:2970,0:3]
estimate_exp19=estimate_exp19[0:1350,0:3]

gt_exp1=numpy.loadtxt("dataset/exp1/ground_truth.txt")
gt_exp2=numpy.loadtxt("dataset/exp2/ground_truth.txt")
gt_exp3=numpy.loadtxt("dataset/exp3/ground_truth.txt")
gt_exp4=numpy.loadtxt("dataset/exp4/ground_truth.txt")
gt_exp5=numpy.loadtxt("dataset/exp5/ground_truth.txt")
gt_exp6=numpy.loadtxt("dataset/exp6/ground_truth.txt")
gt_exp7=numpy.loadtxt("dataset/exp7/ground_truth.txt")
gt_exp8=numpy.loadtxt("dataset/exp8/ground_truth.txt")
gt_exp9=numpy.loadtxt("dataset/exp9/ground_truth.txt")
gt_exp10=numpy.loadtxt("dataset/exp10/ground_truth.txt")
gt_exp11=numpy.loadtxt("dataset/exp11/ground_truth.txt")
gt_exp12=numpy.loadtxt("dataset/exp12/ground_truth.txt")
gt_exp13=numpy.loadtxt("dataset/exp13/ground_truth.txt")
gt_exp14=numpy.loadtxt("dataset/exp14/ground_truth.txt")
gt_exp15=numpy.loadtxt("dataset/exp15/ground_truth.txt")
gt_exp16=numpy.loadtxt("dataset/exp16/ground_truth.txt")
gt_exp17=numpy.loadtxt("dataset/exp17/ground_truth.csv", delimiter=";")
gt_exp18=numpy.loadtxt("dataset/exp18/ground_truth.csv", delimiter=";")
gt_exp19=numpy.loadtxt("dataset/exp19/ground_truth.csv", delimiter=";")

gt_exp1=gt_exp1[0:2200,1:4]
gt_exp2=gt_exp2[0:1300,1:4]
gt_exp3=gt_exp3[0:1100,1:4]
gt_exp4=gt_exp4[0:1100,1:4]
gt_exp5=gt_exp5[0:1200,1:4]
gt_exp6=gt_exp6[0:1700,1:4]
gt_exp7=gt_exp7[0:2100,1:4]
gt_exp8=gt_exp8[0:2100,1:4]
gt_exp9=gt_exp9[0:4000,1:4]
gt_exp10=gt_exp10[0:2700,1:4]
gt_exp11=gt_exp11[0:1700,1:4]
gt_exp12=gt_exp12[0:700,1:4]
gt_exp13=gt_exp13[0:1000,1:4]
gt_exp14=gt_exp14[0:1500,1:4]
gt_exp15=gt_exp15[0:2300,1:4]
gt_exp16=gt_exp16[0:1200,1:4]
gt_exp17=gt_exp17[0:420,0:3]
gt_exp18=gt_exp18[0:2970,0:3]
gt_exp19=gt_exp19[0:1350,0:3]

#gt_exp1_1=gt_exp1_1[0:1800,1:4]


all_estimates=estimate_exp1
all_estimates=numpy.concatenate((all_estimates, estimate_exp2))
all_estimates=numpy.concatenate((all_estimates, estimate_exp3))
all_estimates=numpy.concatenate((all_estimates, estimate_exp4))
all_estimates=numpy.concatenate((all_estimates, estimate_exp5))
all_estimates=numpy.concatenate((all_estimates, estimate_exp6))
all_estimates=numpy.concatenate((all_estimates, estimate_exp7))
all_estimates=numpy.concatenate((all_estimates, estimate_exp8))
all_estimates=numpy.concatenate((all_estimates, estimate_exp9))
all_estimates=numpy.concatenate((all_estimates, estimate_exp10))
all_estimates=numpy.concatenate((all_estimates, estimate_exp11))
all_estimates=numpy.concatenate((all_estimates, estimate_exp12))
all_estimates=numpy.concatenate((all_estimates, estimate_exp13))
all_estimates=numpy.concatenate((all_estimates, estimate_exp14))
all_estimates=numpy.concatenate((all_estimates, estimate_exp15))
all_estimates=numpy.concatenate((all_estimates, estimate_exp16))
all_estimates=numpy.concatenate((all_estimates, estimate_exp17))
all_estimates=numpy.concatenate((all_estimates, estimate_exp18))
all_estimates=numpy.concatenate((all_estimates, estimate_exp19))

all_gts=gt_exp1
all_gts=numpy.concatenate((all_gts, gt_exp2))
all_gts=numpy.concatenate((all_gts, gt_exp3))
all_gts=numpy.concatenate((all_gts, gt_exp4))
all_gts=numpy.concatenate((all_gts, gt_exp5))
all_gts=numpy.concatenate((all_gts, gt_exp6))
all_gts=numpy.concatenate((all_gts, gt_exp7))
all_gts=numpy.concatenate((all_gts, gt_exp8))
all_gts=numpy.concatenate((all_gts, gt_exp9))
all_gts=numpy.concatenate((all_gts, gt_exp10))
all_gts=numpy.concatenate((all_gts, gt_exp11))
all_gts=numpy.concatenate((all_gts, gt_exp12))
all_gts=numpy.concatenate((all_gts, gt_exp13))
all_gts=numpy.concatenate((all_gts, gt_exp14))
all_gts=numpy.concatenate((all_gts, gt_exp15))
all_gts=numpy.concatenate((all_gts, gt_exp16))
all_gts=numpy.concatenate((all_gts, gt_exp17))
all_gts=numpy.concatenate((all_gts, gt_exp18))
all_gts=numpy.concatenate((all_gts, gt_exp19))

estimate_scaler = MinMaxScaler(feature_range=(0.05, 0.95))
gt_scaler       = MinMaxScaler(feature_range=(0.05, 0.95))

estimate_scaler.fit(all_estimates);
gt_scaler.fit(all_gts); 

estimate_exp1_scaled=estimate_scaler.transform(estimate_exp1) 
estimate_exp2_scaled=estimate_scaler.transform(estimate_exp2) 
estimate_exp3_scaled=estimate_scaler.transform(estimate_exp3) 
estimate_exp4_scaled=estimate_scaler.transform(estimate_exp4) 
estimate_exp5_scaled=estimate_scaler.transform(estimate_exp5) 
estimate_exp6_scaled=estimate_scaler.transform(estimate_exp6) 
estimate_exp7_scaled=estimate_scaler.transform(estimate_exp7) 
estimate_exp8_scaled=estimate_scaler.transform(estimate_exp8) 
estimate_exp9_scaled=estimate_scaler.transform(estimate_exp9) 
estimate_exp10_scaled=estimate_scaler.transform(estimate_exp10) 
estimate_exp11_scaled=estimate_scaler.transform(estimate_exp11) 
estimate_exp12_scaled=estimate_scaler.transform(estimate_exp12) 
estimate_exp13_scaled=estimate_scaler.transform(estimate_exp13) 
estimate_exp14_scaled=estimate_scaler.transform(estimate_exp14) 
estimate_exp15_scaled=estimate_scaler.transform(estimate_exp15) 
estimate_exp16_scaled=estimate_scaler.transform(estimate_exp16) 
estimate_exp17_scaled=estimate_scaler.transform(estimate_exp17) 
estimate_exp18_scaled=estimate_scaler.transform(estimate_exp18) 
estimate_exp19_scaled=estimate_scaler.transform(estimate_exp19) 

gt_exp1_scaled=gt_scaler.transform(gt_exp1) 
gt_exp2_scaled=gt_scaler.transform(gt_exp2) 
gt_exp3_scaled=gt_scaler.transform(gt_exp3) 
gt_exp4_scaled=gt_scaler.transform(gt_exp4) 
gt_exp5_scaled=gt_scaler.transform(gt_exp5) 
gt_exp6_scaled=gt_scaler.transform(gt_exp6) 
gt_exp7_scaled=gt_scaler.transform(gt_exp7) 
gt_exp8_scaled=gt_scaler.transform(gt_exp8) 
gt_exp9_scaled=gt_scaler.transform(gt_exp9) 
gt_exp10_scaled=gt_scaler.transform(gt_exp10) 
gt_exp11_scaled=gt_scaler.transform(gt_exp11) 
gt_exp12_scaled=gt_scaler.transform(gt_exp12) 
gt_exp13_scaled=gt_scaler.transform(gt_exp13) 
gt_exp14_scaled=gt_scaler.transform(gt_exp14) 
gt_exp15_scaled=gt_scaler.transform(gt_exp15) 
gt_exp16_scaled=gt_scaler.transform(gt_exp16) 
gt_exp17_scaled=gt_scaler.transform(gt_exp17) 
gt_exp18_scaled=gt_scaler.transform(gt_exp18) 
gt_exp19_scaled=gt_scaler.transform(gt_exp19) 

#dump(estimate_scaler, 'cont1_new_data/results/estimate_scaler.bin', compress=True)
#dump(gt_scaler, 'cont1_new_data/results/gt_scaler.bin', compress=True)


exp1_sequences, exp1_target_poses, exp1_input_poses = convert_trajectory_to_sequence(seq_length, estimate_exp1_scaled, gt_exp1_scaled)
exp2_sequences, exp2_target_poses, exp2_input_poses = convert_trajectory_to_sequence(seq_length, estimate_exp2_scaled, gt_exp2_scaled)
exp3_sequences, exp3_target_poses, exp3_input_poses = convert_trajectory_to_sequence(seq_length, estimate_exp3_scaled, gt_exp3_scaled)
exp4_sequences, exp4_target_poses, exp4_input_poses = convert_trajectory_to_sequence(seq_length, estimate_exp4_scaled, gt_exp4_scaled)
exp5_sequences, exp5_target_poses, exp5_input_poses = convert_trajectory_to_sequence(seq_length, estimate_exp5_scaled, gt_exp5_scaled)
exp6_sequences, exp6_target_poses, exp6_input_poses = convert_trajectory_to_sequence(seq_length, estimate_exp6_scaled, gt_exp6_scaled)
exp7_sequences, exp7_target_poses, exp7_input_poses = convert_trajectory_to_sequence(seq_length, estimate_exp7_scaled, gt_exp7_scaled)
exp8_sequences, exp8_target_poses, exp8_input_poses = convert_trajectory_to_sequence(seq_length, estimate_exp8_scaled, gt_exp8_scaled)
exp9_sequences, exp9_target_poses, exp9_input_poses = convert_trajectory_to_sequence(seq_length, estimate_exp9_scaled, gt_exp9_scaled)
exp10_sequences, exp10_target_poses, exp10_input_poses = convert_trajectory_to_sequence(seq_length, estimate_exp10_scaled, gt_exp10_scaled)
exp11_sequences, exp11_target_poses, exp11_input_poses = convert_trajectory_to_sequence(seq_length, estimate_exp11_scaled, gt_exp11_scaled)
exp12_sequences, exp12_target_poses, exp12_input_poses = convert_trajectory_to_sequence(seq_length, estimate_exp12_scaled, gt_exp12_scaled)
exp13_sequences, exp13_target_poses, exp13_input_poses = convert_trajectory_to_sequence(seq_length, estimate_exp13_scaled, gt_exp13_scaled)  
exp14_sequences, exp14_target_poses, exp14_input_poses = convert_trajectory_to_sequence(seq_length, estimate_exp14_scaled, gt_exp14_scaled)
exp15_sequences, exp15_target_poses, exp15_input_poses = convert_trajectory_to_sequence(seq_length, estimate_exp15_scaled, gt_exp15_scaled)
exp16_sequences, exp16_target_poses, exp16_input_poses = convert_trajectory_to_sequence(seq_length, estimate_exp16_scaled, gt_exp16_scaled)  
exp17_sequences, exp17_target_poses, exp17_input_poses = convert_trajectory_to_sequence(seq_length, estimate_exp17_scaled, gt_exp17_scaled)
exp18_sequences, exp18_target_poses, exp18_input_poses = convert_trajectory_to_sequence(seq_length, estimate_exp18_scaled, gt_exp18_scaled)
exp19_sequences, exp19_target_poses, exp19_input_poses = convert_trajectory_to_sequence(seq_length, estimate_exp19_scaled, gt_exp19_scaled)  


training_sequences=exp1_sequences; 
training_sequences=numpy.concatenate((training_sequences, exp2_sequences));
training_sequences=numpy.concatenate((training_sequences, exp3_sequences));
training_sequences=numpy.concatenate((training_sequences, exp4_sequences));
training_sequences=numpy.concatenate((training_sequences, exp5_sequences));
training_sequences=numpy.concatenate((training_sequences, exp6_sequences));
training_sequences=numpy.concatenate((training_sequences, exp7_sequences));
training_sequences=numpy.concatenate((training_sequences, exp8_sequences));
training_sequences=numpy.concatenate((training_sequences, exp9_sequences));
training_sequences=numpy.concatenate((training_sequences, exp10_sequences));
training_sequences=numpy.concatenate((training_sequences, exp11_sequences));
training_sequences=numpy.concatenate((training_sequences, exp12_sequences));
training_sequences=numpy.concatenate((training_sequences, exp13_sequences));
training_sequences=numpy.concatenate((training_sequences, exp14_sequences));
training_sequences=numpy.concatenate((training_sequences, exp15_sequences));
training_sequences=numpy.concatenate((training_sequences, exp16_sequences));
training_sequences=numpy.concatenate((training_sequences, exp17_sequences));
training_sequences=numpy.concatenate((training_sequences, exp18_sequences));
training_sequences=numpy.concatenate((training_sequences, exp19_sequences));


target_poses=exp1_target_poses; 
target_poses=numpy.concatenate((target_poses, exp2_target_poses));
target_poses=numpy.concatenate((target_poses, exp3_target_poses));
target_poses=numpy.concatenate((target_poses, exp4_target_poses));
target_poses=numpy.concatenate((target_poses, exp5_target_poses));
target_poses=numpy.concatenate((target_poses, exp6_target_poses));
target_poses=numpy.concatenate((target_poses, exp7_target_poses));
target_poses=numpy.concatenate((target_poses, exp8_target_poses));
target_poses=numpy.concatenate((target_poses, exp9_target_poses));
target_poses=numpy.concatenate((target_poses, exp10_target_poses));
target_poses=numpy.concatenate((target_poses, exp11_target_poses));
target_poses=numpy.concatenate((target_poses, exp12_target_poses));
target_poses=numpy.concatenate((target_poses, exp13_target_poses));
target_poses=numpy.concatenate((target_poses, exp14_target_poses));
target_poses=numpy.concatenate((target_poses, exp15_target_poses));
target_poses=numpy.concatenate((target_poses, exp16_target_poses));
target_poses=numpy.concatenate((target_poses, exp17_target_poses));
target_poses=numpy.concatenate((target_poses, exp18_target_poses));
target_poses=numpy.concatenate((target_poses, exp19_target_poses));


input_poses=exp1_input_poses; 
input_poses=numpy.concatenate((input_poses, exp2_input_poses));
input_poses=numpy.concatenate((input_poses, exp3_input_poses));
input_poses=numpy.concatenate((input_poses, exp4_input_poses));
input_poses=numpy.concatenate((input_poses, exp5_input_poses));
input_poses=numpy.concatenate((input_poses, exp6_input_poses));
input_poses=numpy.concatenate((input_poses, exp7_input_poses));
input_poses=numpy.concatenate((input_poses, exp8_input_poses));
input_poses=numpy.concatenate((input_poses, exp9_input_poses));
input_poses=numpy.concatenate((input_poses, exp10_input_poses));
input_poses=numpy.concatenate((input_poses, exp11_input_poses));
input_poses=numpy.concatenate((input_poses, exp12_input_poses));
input_poses=numpy.concatenate((input_poses, exp13_input_poses));
input_poses=numpy.concatenate((input_poses, exp14_input_poses));
input_poses=numpy.concatenate((input_poses, exp15_input_poses));
input_poses=numpy.concatenate((input_poses, exp16_input_poses));
input_poses=numpy.concatenate((input_poses, exp17_input_poses));
input_poses=numpy.concatenate((input_poses, exp18_input_poses));
input_poses=numpy.concatenate((input_poses, exp19_input_poses));

X_train, X_test_validate, Y_train, Y_test_validate, input_poses_train, input_poses_test_validate=train_test_split(training_sequences, target_poses, input_poses, test_size=0.2)
X_test, X_validate, Y_test, Y_validate, input_poses_test, input_poses_validate=train_test_split(X_test_validate, Y_test_validate, input_poses_test_validate, test_size=0.5)

tf.keras.backend.clear_session()
model=tf.keras.Sequential()

model.add(tf.keras.layers.LSTM(layer_size, input_shape=(seq_length, 3), return_sequences=True))
model.add(tf.keras.layers.Dropout(0.1))
model.add(tf.keras.layers.LSTM(layer_size, input_shape=(seq_length, 3), return_sequences=False))
model.add(tf.keras.layers.Dropout(0.1))
model.add(tf.keras.layers.Dense(3, activation="sigmoid"))


model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['accuracy'])
history=model.fit(X_train, Y_train, validation_data=(X_validate, Y_validate), epochs=1000, batch_size=100, verbose=2, shuffle=True)
model.save('dataset/dataset_random_train_validate_test/model.hdf5')

exp1_output_scaled = model.predict(exp1_sequences, batch_size=1)
exp2_output_scaled = model.predict(exp2_sequences, batch_size=1)
exp3_output_scaled = model.predict(exp3_sequences, batch_size=1)
exp4_output_scaled = model.predict(exp4_sequences, batch_size=1)
exp5_output_scaled = model.predict(exp5_sequences, batch_size=1)
exp6_output_scaled = model.predict(exp6_sequences, batch_size=1)
exp7_output_scaled = model.predict(exp7_sequences, batch_size=1)
exp8_output_scaled = model.predict(exp8_sequences, batch_size=1)
exp9_output_scaled = model.predict(exp9_sequences, batch_size=1)
exp10_output_scaled = model.predict(exp10_sequences, batch_size=1)
exp11_output_scaled = model.predict(exp11_sequences, batch_size=1)
exp12_output_scaled = model.predict(exp12_sequences, batch_size=1)
exp13_output_scaled = model.predict(exp13_sequences, batch_size=1)
exp14_output_scaled = model.predict(exp14_sequences, batch_size=1)
exp15_output_scaled = model.predict(exp15_sequences, batch_size=1)
exp16_output_scaled = model.predict(exp16_sequences, batch_size=1)
exp17_output_scaled = model.predict(exp17_sequences, batch_size=1)
exp18_output_scaled = model.predict(exp18_sequences, batch_size=1)
exp19_output_scaled = model.predict(exp19_sequences, batch_size=1)
training_output_scaled=model.predict(X_train, batch_size=1)
validation_output_scaled=model.predict(X_validate, batch_size=1)
testing_output_scaled=model.predict(X_test, batch_size=1)

exp1_output=gt_scaler.inverse_transform(exp1_output_scaled)
exp2_output=gt_scaler.inverse_transform(exp2_output_scaled)
exp3_output=gt_scaler.inverse_transform(exp3_output_scaled)
exp4_output=gt_scaler.inverse_transform(exp4_output_scaled)
exp5_output=gt_scaler.inverse_transform(exp5_output_scaled)
exp6_output=gt_scaler.inverse_transform(exp6_output_scaled)
exp7_output=gt_scaler.inverse_transform(exp7_output_scaled)
exp8_output=gt_scaler.inverse_transform(exp8_output_scaled)
exp9_output=gt_scaler.inverse_transform(exp9_output_scaled)
exp10_output=gt_scaler.inverse_transform(exp10_output_scaled)
exp11_output=gt_scaler.inverse_transform(exp11_output_scaled)
exp12_output=gt_scaler.inverse_transform(exp12_output_scaled)
exp13_output=gt_scaler.inverse_transform(exp13_output_scaled)
exp14_output=gt_scaler.inverse_transform(exp14_output_scaled)
exp15_output=gt_scaler.inverse_transform(exp15_output_scaled)
exp16_output=gt_scaler.inverse_transform(exp16_output_scaled)
exp17_output=gt_scaler.inverse_transform(exp17_output_scaled)
exp18_output=gt_scaler.inverse_transform(exp18_output_scaled)
exp19_output=gt_scaler.inverse_transform(exp19_output_scaled)
training_output=gt_scaler.inverse_transform(training_output_scaled)
validation_output=gt_scaler.inverse_transform(validation_output_scaled)
testing_output=gt_scaler.inverse_transform(testing_output_scaled)

write_file (exp1_output, 'exp1_output.csv')
write_file (exp2_output, 'exp2_output.csv')
write_file (exp3_output, 'exp3_output.csv')
write_file (exp4_output, 'exp4_output.csv')
write_file (exp5_output, 'exp5_output.csv')
write_file (exp6_output, 'exp6_output.csv')
write_file (exp7_output, 'exp7_output.csv')
write_file (exp8_output, 'exp8_output.csv')
write_file (exp9_output, 'exp9_output.csv')
write_file (exp10_output, 'exp10_output.csv')
write_file (exp11_output, 'exp11_output.csv')
write_file (exp12_output, 'exp12_output.csv')
write_file (exp13_output, 'exp13_output.csv')
write_file (exp14_output, 'exp14_output.csv')
write_file (exp15_output, 'exp15_output.csv')
write_file (exp16_output, 'exp16_output.csv')
write_file (exp17_output, 'exp17_output.csv')
write_file (exp18_output, 'exp18_output.csv')
write_file (exp19_output, 'exp19_output.csv')
write_file ([history.history['accuracy']], 'acc_history.csv')
write_file ([history.history['loss']], 'loss_history.csv')
write_file ([history.history['val_loss']], 'val_loss_history.csv')
write_file ([history.history['val_accuracy']], 'val_acc_history.csv')
write_file (input_poses_train, 'training_input.csv')
write_file (Y_train, 'training_gt.csv')
write_file (training_output, 'training_output.csv')

write_file (input_poses_validate, 'validation_input.csv')
write_file (Y_validate, 'validation_gt.csv')
write_file (validation_output, 'validation_output.csv')

write_file (input_poses_test, 'testing_input.csv')
write_file (Y_test, 'testing_gt.csv')
write_file (testing_output, 'testing_output.csv')
