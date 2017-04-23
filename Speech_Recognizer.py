import tflearn
import speech_data

#hyper paramters aka tuning nobs
learnin_rate = 0.0001 #greater means faster training, lower is higher accuracy
training_iters = 300000 #number of steps to train for

batch = word_batch = speech_data.mfcc_batch_gernator(64)
X, Y = next(batch)
trainX, trainY = X, Y
textX, testY, X, Y

#call recurrent neural network
net = tflearn.input_data([None, 20, 80]) #width is number of features extraceted and height is max length of each utterance
net = tflearn.lstm(net, 128, dropout=0.8) 
net = tflearn.fully_connected(net, 10, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=learning_rate, loss='categorical_crossentropy')

model = tflearn.DNN(net, tensorboard_verbose=0)
while 1:
	model.fit(trainX, trainY, n_epoch=10, validation_set=(testX, testY), show_metric=True,
			batch_size=64)