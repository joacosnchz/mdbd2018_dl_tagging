import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import tensorflow as tf
import numpy as np
import nltk
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tags_dataset import create_feature_sets_and_labels
from tags_model import TagsModel
import utils
import msgpack
import tags_worddict
from datetime import datetime

FILE_PATH = os.path.dirname(os.path.realpath(__file__)) # path where the input data is stored
BATCH_SIZE = int(sys.argv[1]) # amount of posts to take by time sys.argv
HM_EPOCHS = int(sys.argv[2]) # how many epochs per training
IS_TRAINING = bool(sys.argv[3]) # are we training or using the model?
N_CLASSES = int(sys.argv[4]) # how many neurons in the output layer
TRAINING_INDEX = int(sys.argv[5])

model = TagsModel(N_CLASSES)

(train_x, train_y), (test_x, test_y), indexes = utils.read_data(FILE_PATH)

long_training = len(train_x[0]) # 86

x = tf.placeholder('float', [None, long_training], name='x')
y = tf.placeholder('float', name='y')

# es necesario inicializar las variables antes del saver para que tenga algo que guardar
l1, ol = model.initialize_variables(long_training)

saver = tf.train.Saver()

def train_neural_network(x, hm_epochs, batch_size):
    prediction = model.predict(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    loss = open('./output/loss.txt', 'a')
    trainings = open('./output/training.txt', 'a')
    
    # feed forward + backpropagation = epoch
    
    initial_time = datetime.now()
    print('Starting training...')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for epoch in range(hm_epochs):
            epoch_loss = 0
            
            
            i = 0
            while i < len(train_x):
                start = i
                end = i + batch_size

                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])

                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                epoch_loss += c
                i += batch_size

            #Log the data from the epoch in console and file
            loss.write(str(TRAINING_INDEX) + '\t' + str(HM_EPOCHS) + '\t' + str(epoch) + '\t' + str(BATCH_SIZE) + '\t' + str(epoch_loss) + '\n')
            saver.save(sess, './output/model_trains' + '_epochs_' + str(HM_EPOCHS) + '_batch_' + str(BATCH_SIZE) + '/model.ckpt')
            print('Epoch', epoch+1, 'completed out of', hm_epochs, 'loss:', epoch_loss)

        end = datetime.now()
        time_diff = end - initial_time
        
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        # Splitting the accuracy calculation in order to save memory
        test_size = int(len(test_x) * .5)
        acc1 = accuracy.eval({x: test_x[:test_size], y: test_y[:test_size]})
        acc2 = accuracy.eval({x: test_x[test_size:], y: test_y[test_size:]})
        acc = ((acc1*test_size) + (acc2*test_size))/ (2*test_size)
        print('Accuracy:', acc)

        #Log the build's accuracy
        trainings.write(str(TRAINING_INDEX) + '\t'+ str(time_diff) + '\t' + str(acc) + '\n')
       
        loss.close()
        trainings.close()
        


def feed_forward(test_x):
    '''Takes vector representation of a text. Runs the model and returns the most accurate label or category.
    '''
    prediction = model.predict(x)
    result = (sess.run(tf.argmax(prediction.eval(feed_dict={x:test_x}),1)))
    print("Data predicted saccessfully")
    return (result)


def generate_confusion_matrix(pred, test_y, plot=False):
    '''Generates the confusion matrix of the tested data
    '''
    heatmap_data = tf.confusion_matrix(labels=test_y, predictions=pred)

    sess = tf.Session()
    with sess.as_default():
        heatmap_matrix = sess.run(heatmap_data)

    df = pd.DataFrame(heatmap_matrix, index=indexes, columns=indexes)
    print("Heatmap generated saccessfully")

    #Plot the matrix using seaborn 
    utils.plot_confusion_matrix(df, plot)




if __name__ == '__main__':
    if IS_TRAINING:
        train_neural_network(x, HM_EPOCHS, BATCH_SIZE)
        pred = feed_forward(test_x)
        generate_confusion_matrix(pred, test_y, True)


    else:
        print('Coffee')
        utils.use_neural_network('we got two of these for our office one has specialties menu item the other does not the specialties menu item is mentioned in the user guide but not how to enable or disable it in the picture below it is the lower right grid item it is just blank missing from one machine')
        utils.use_neural_network('i want to understand how i can make a better cup of coffee so i recently purchased a wilfa grinder however after reading the instruction manual it says that the blades and bean cup cannot be submerged in water only wiped clean this means that you are never really going to get it spotlessly clean like you can with a manual grinder')

        print('Vi')
        utils.use_neural_network('i am looking to lazily start up plug in when the user starts using vim this is to save resources when user may start up lots of vims and then not interact with it')  
        utils.use_neural_network('the problem is that i cant do anything inside vim until i close the powershell window which somehow defeats the purpose how can i tell vim to let go of the opened powershell so i could make changes in the file open in vim')

