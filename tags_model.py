import tensorflow as tf

class TagsModel:
    def __init__(self):
        self.n_nodes_hl1 = 500
        self.n_nodes_hl2 = 500
        self.n_nodes_hl3 = 500
        self.n_classes = 50

    def initialize_variables(self, long_training):
        self.hidden_1_layer = {'weights':tf.Variable(tf.random_normal([long_training, self.n_nodes_hl1])),
                          'biases':tf.Variable(tf.random_normal([self.n_nodes_hl1]))}

        self.hidden_2_layer = {'weights':tf.Variable(tf.random_normal([self.n_nodes_hl1, self.n_nodes_hl2])),
                          'biases':tf.Variable(tf.random_normal([self.n_nodes_hl2]))}

        self.hidden_3_layer = {'weights':tf.Variable(tf.random_normal([self.n_nodes_hl2, self.n_nodes_hl3])),
                          'biases':tf.Variable(tf.random_normal([self.n_nodes_hl3]))}

        self.output_layer = {'weights':tf.Variable(tf.random_normal([self.n_nodes_hl3, self.n_classes])),
                        'biases':tf.Variable(tf.random_normal([self.n_classes]))}

        return self.hidden_1_layer, self.hidden_2_layer, self.hidden_3_layer, self.output_layer

    def predict(self, data):
        l1 = self.layer_1(data)

        l2 = self.layer_2(l1)

        l3 = self.layer_3(l2)

        output = tf.matmul(l3, self.output_layer['weights']) + self.output_layer['biases']

        return output

    def layer_1(self, input_data):
        # y = (x * weights) + biases
        l1 = tf.add(tf.matmul(input_data, self.hidden_1_layer['weights']), self.hidden_1_layer['biases'])
        l1 = tf.nn.relu(l1) # f(x) = max(0, x)

        return l1

    def layer_2(self, input_data):
        l2 = tf.add(tf.matmul(input_data, self.hidden_2_layer['weights']), self.hidden_2_layer['biases'])
        l2 = tf.nn.relu(l2)

        return l2

    def layer_3(self, input_data):
        l3 = tf.add(tf.matmul(input_data, self.hidden_3_layer['weights']), self.hidden_3_layer['biases'])
        l3 = tf.nn.relu(l3)

        return l3

