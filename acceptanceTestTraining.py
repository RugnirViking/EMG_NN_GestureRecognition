import random
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

print("Tensorflow Version: "+tf.__version__)
print("WELCOME TO HAL. PLEASE KEEP ALL AMRS AND LEGS INSIDE OF THE RIDE AT ALL TIMES.")

class_names = ['Claw','Fk','Ok','Pointing','Resting','Thumbsup']
def getDataFromCategoryName( name ):
    "gets data from the folder called categoryname"
    f=open("datasets/acceptance/"+name+"/!output.txt", "r")
    a = []
    if f.mode == 'r':
        
        for x in f:
            # put each line of the file into an array
            currentline = x.split(",")
            n0 = int(currentline[0])
            n1 = int(currentline[1])
            n2 = int(currentline[2])
            n3 = int(currentline[3])
            n4 = int(currentline[4])
            n5 = int(currentline[5])
            n6 = int(currentline[6])
            n7 = int(currentline[7])
            if (n0==0 and n1==0 and n2==0 and n3==0 and n4==0 and n5==0 and n6==0 and n7==0):
                continue
            a.append([n0,n1,n2,n3,n4,n5,n6,n7])
        
    else:
        print ("ERROR: File read failed")
    
    b = []
    for x in range(0, len(a)-50):
        b.append(a[x:x+50])
    return b
claw_test_data = getDataFromCategoryName("claw")
fk_test_data = getDataFromCategoryName("fk-a-uuu")
ok_test_data = getDataFromCategoryName("ok")
pointing_test_data = getDataFromCategoryName("pointing")
resting_test_data = getDataFromCategoryName("resting")
thumbup_test_data = getDataFromCategoryName("thumbup")

overall_ordered_test_data =  thumbup_test_data + resting_test_data
overall_ordered_test_labels=[]
overall_ordered_test_labels = overall_ordered_test_labels + [5]*len(thumbup_test_data) + [4]*len(resting_test_data)

#overall_ordered_test_data = claw_test_data + fk_test_data + ok_test_data + pointing_test_data + resting_test_data + thumbup_test_data
#overall_ordered_test_labels=[]
#overall_ordered_test_labels = overall_ordered_test_labels + [0]*len(claw_test_data) + [1]*len(fk_test_data) + [2]*len(ok_test_data) + [3]*len(pointing_test_data) + [4]*len(resting_test_data) + [5]*len(thumbup_test_data)

combined = list(zip(overall_ordered_test_data, overall_ordered_test_labels))
random.shuffle(combined)

overall_ordered_test_data[:], overall_ordered_test_labels[:] = zip(*combined)

print(len(overall_ordered_test_labels))
#overall_ordered_test_labels = np.ones(len(claw_test_data))*0+np.ones(len(fk_test_data))*1+np.ones(len(ok_test_data))*2+np.ones(len(pointing_test_data))*3+np.ones(len(resting_test_data))*4+np.ones(len(thumbup_test_data))*5

train_data = int((len(overall_ordered_test_data)*9)/10)
training_data = overall_ordered_test_data[:train_data]

testing_data = overall_ordered_test_data[train_data:]

training_labels = overall_ordered_test_labels[:train_data]
testing_labels = overall_ordered_test_labels[train_data:]
# fashion_mnist = keras.datasets.fashion_mnist
# (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
# print(train_images[0])
# print(training_data[0])
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(50, 8)),
    #keras.layers.Dense(512, input_shape=(100,8), activation=tf.nn.relu),
    keras.layers.Dense(800, activation=tf.nn.relu),
    keras.layers.Dense(800, activation=tf.nn.relu),
    keras.layers.Dense(6, activation=tf.nn.softmax)
])
model.compile(optimizer=tf.train.AdamOptimizer(),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])
# model.compile(optimizer='rmsprop',
              # loss='categorical_crossentropy',
              # metrics=['accuracy'])

history = model.fit(np.array(training_data), training_labels, epochs=7,batch_size=512,
                    validation_data=(np.array(testing_data), testing_labels),
                    verbose=1)
test_loss, test_acc = model.evaluate(np.array(testing_data), testing_labels)
print('Test accuracy:', test_acc)
history_dict = history.history

predictions = model.predict(np.array(testing_data))




def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  
  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
  
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(6), predictions_array, color="#777777")
  plt.ylim([0, 1]) 
  predicted_label = np.argmax(predictions_array)
 
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')
predictions = model.predict(np.array(testing_data))
num_rows = 10
num_cols = 6
num_images = len(predictions)
plt.figure(figsize=(2*num_cols, 1*num_rows))

class_names = ['Claw','Fk','Ok','Pointing','Resting','Thumbsup']

claw_success = [0,0]
fk_success = [0,0]
ok_success = [0,0]
point_success = [0,0]
rest_success = [0,0]
thumbsup_success = [0,0]

for i in range(num_images):
  npPredictions = np.array(predictions)
  predictions_this = npPredictions[i]
  true_label = testing_labels[i]
  predicted_label = np.argmax(predictions_this)
  if predicted_label == true_label:
    
    if predicted_label == 0:
        claw_success[0]+=1
    if predicted_label == 1:
        fk_success[0]+=1
    if predicted_label == 2:
        ok_success[0]+=1
    if predicted_label == 3:
        point_success[0]+=1
    if predicted_label == 4:
        rest_success[0]+=1
    if predicted_label == 5:
        thumbsup_success[0]+=1
  else:
    if predicted_label == 0:
        claw_success[1]+=1
    if predicted_label == 1:
        fk_success[1]+=1
    if predicted_label == 2:
        ok_success[1]+=1
    if predicted_label == 3:
        point_success[1]+=1
    if predicted_label == 4:
        rest_success[1]+=1
    if predicted_label == 5:
        thumbsup_success[1]+=1

if claw_success[0]>0:
  print("Claw Success Rate: "+str(round(claw_success[0]/(claw_success[0]+claw_success[1]),4)*100)+"% Raw numbers: "+str(claw_success[0])+" & Fails: "+str(claw_success[1]))
if fk_success[0]>0:
  print("Fuck-a-youuu Success Rate: "+str(round(fk_success[0]/(fk_success[0]+fk_success[1]),4)*100)+"% Raw numbers: "+str(fk_success[0])+" & Fails: "+str(fk_success[1]))
if ok_success[0]>0:
  print("Ok Success Rate: "+str(round(ok_success[0]/(ok_success[0]+ok_success[1]),4)*100)+"% Raw numbers: "+str(ok_success[0])+" & Fails: "+str(ok_success[1]))
if point_success[0]>0:
  print("Pointing Success Rate: "+str(round(point_success[0]/(point_success[0]+point_success[1]),4)*100)+"% Raw numbers: "+str(point_success[0])+" & Fails: "+str(point_success[1]))
if rest_success[0]>0:
  print("Resting Success Rate: "+str(round(rest_success[0]/(rest_success[0]+rest_success[1]),4)*100)+"% Raw numbers: "+str(rest_success[0])+" & Fails: "+str(rest_success[1]))
if thumbsup_success[0]>0:
  print("Thumbsup Success Rate: "+str(round(thumbsup_success[0]/(thumbsup_success[0]+thumbsup_success[1]),4)*100)+"% Raw numbers: "+str(thumbsup_success[0])+" & Fails: "+str(thumbsup_success[1]))

model.save("model/trainedAcceptanceTestModel.h5");









