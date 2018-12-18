from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import socket
import socket
import numpy as np
import tensorflow as tf
import argparse
import sys
import io
import time

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
model_file = "../output/output_graph.pb"
def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph
def createArrayFromRecievedString(str_recv):
  splitString = str_recv.split("|")
  arrayA = []
  for x in range(0, len(splitString)-1):
    arrayA.append(float(splitString[x]))
  arrayB = np.reshape(arrayA, (-1, 8))
  arrayC = []
  arrayC.append(arrayB)
  return np.array(arrayC)
s.bind(('localhost', 3333))

s.listen(5)
flag = 0
graph = load_graph(model_file)
input_name = "import/flatten_input"
output_name = "import/k2tfout_0"
input_operation = graph.get_operation_by_name(input_name);
output_operation = graph.get_operation_by_name(output_name);
lastResult='Resting'
eggList = ['Resting','Resting','Resting','Resting','Resting','Resting','Resting','Resting','Resting','Resting']
with tf.Session(graph=graph) as sess:
    while True:
        connect, addr = s.accept()
        print("Connection Address:" + str(addr))
    
        while True:
            str_recv, temp = connect.recvfrom(16384)
            predictionArray = createArrayFromRecievedString(str_recv.decode("ascii"))
            
            
            results = sess.run(output_operation.outputs[0],{input_operation.outputs[0]: predictionArray})
                
            results = np.squeeze(results)
            top_k = results.argsort()[-1:][::-1]
            labels = ['Claw','Fk','Ok','Pointing','Resting','Thumbsup']
            
            for i in top_k:
                
                if (results[i]>0.6):
                    curLabel=labels[i]
                    eggList.insert(0, curLabel)
                    eggList.pop()
                    allSame=True
                    for x in range(0,len(eggList)):
                        if eggList[x]!=curLabel:
                            allSame=False
                    if allSame:
                        # send the current one
                        lastResult = curLabel
                        str_return = curLabel+","+str(results[i])
                        connect.sendto(bytes(str_return, 'utf-8'), addr)
                    else:
                        # send the old one
                        str_return = lastResult+","+str(results[i])
                        connect.sendto(bytes(str_return, 'utf-8'), addr)
                else:
                    str_return = lastResult+","+str(results[i])
                    connect.sendto(bytes(str_return, 'utf-8'), addr)
                
            #connect.sendto(bytes("wog", 'utf-8'), addr)
        connect.close()
    
