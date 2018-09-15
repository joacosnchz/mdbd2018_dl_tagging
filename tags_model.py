import tensorflow as tf

class TagsModel:
    def __init__(self, n_classes):
        self.n_nodes_hl1 = 500
        self.n_classes = n_classes

    def initialize_variables(self, long_training):
        self.hidden_1_layer = {'weights':tf.Variable(tf.random_normal([long_training, self.n_nodes_hl1])),
                          'biases':tf.Variable(tf.random_normal([self.n_nodes_hl1]))}

        self.output_layer = {'weights':tf.Variable(tf.random_normal([self.n_nodes_hl1, self.n_classes])),
                        'biases':tf.Variable(tf.random_normal([self.n_classes]))}

        return self.hidden_1_layer, self.output_layer

    def predict(self, data):
        l1 = self.layer_1(data)

        output = tf.matmul(l1, self.output_layer['weights']) + self.output_layer['biases']

        return output

    def layer_1(self, input_data):
        l1 = tf.add(tf.matmul(input_data, self.hidden_1_layer['weights']), self.hidden_1_layer['biases'])
        l1 = tf.nn.relu(l1) # f(x) = max(0, x)

        return l1

