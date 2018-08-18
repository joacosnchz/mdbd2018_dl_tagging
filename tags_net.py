import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import nltk
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tags_dataset import create_feature_sets_and_labels
from tags_model import TagsModel

model = TagsModel()

train_x, train_y, test_x, test_y = create_feature_sets_and_labels()

batch_size = 100
long_training = 86 # len(train_x[0])

x = tf.placeholder('float', [None, long_training], name='x')
y = tf.placeholder('float', name='y')

l1, l2, l3, ol = model.initialize_variables(long_training)

saver = tf.train.Saver()

def train_neural_network(x):
    prediction = model.predict(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    # feed forward + backpropagation = epoch
    hm_epochs = 10
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0

            _, c = sess.run([optimizer, cost], feed_dict={x: train_x, y: train_y})
            epoch_loss += c

            saver.save(sess, './output/model.ckpt')
            print('Epoch', epoch+1, 'completed out of', hm_epochs, 'loss:', epoch_loss)            

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x: test_x, y: test_y}))

def use_neural_network(input_data):
    prediction = model.predict(x)
    lemmatizer = WordNetLemmatizer()
    with open('./input/lexicon.pickle','rb') as f:
        lexicon = pickle.load(f)
        
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, "./output/model.ckpt")
        current_words = word_tokenize(input_data.lower())
        current_words = [lemmatizer.lemmatize(i) for i in current_words]
        features = np.zeros(len(lexicon))

        for word in current_words:
            if word.lower() in lexicon:
                index_value = lexicon.index(word.lower())

                features[index_value] += 1

        features = np.array(list(features))
        result = (sess.run(tf.argmax(prediction.eval(feed_dict={x:[features]}),1)))
        
        if result[0] == 0:
            print('Apache:', input_data)
        elif result[0] == 1:
            print('Php:', input_data)
        elif result[0] == 2:
            print('Pip:', input_data)
        elif result[0] == 3:
            print('Python:', input_data)

train_neural_network(x)
#use_neural_network('pip is not working on my computer')

