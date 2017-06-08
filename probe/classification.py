
import numpy as np

import DNN.mnist_loader as data_loader
import DNN.network as network

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

class SparseClassif:
    def __init__(self, shl, matname, 
                   n_hidden=30, 
                   epochs=30,
                   mini_batch_size=10,
                   eta=.3, 
                 do_linear = False
                ):
        self.shl, self.matname = shl, matname
        self.net = network.Network([self.shl.n_dictionary, n_hidden, 10])
        self.n_hidden = n_hidden 
        self.epochs = epochs
        self.mini_batch_size = mini_batch_size
        self.eta = eta
        self.do_linear = do_linear
        
        training_data, validation_data, test_data = data_loader.load_data()
        self.training_image = training_data[0]
        self.training_supervision = training_data[1]
        self.test_image = test_data[0]
        self.test_supervision = test_data[1]

    def format_data(self):
        if self.do_linear:
            training_vector = (self.dico.dictionary @ self.training_image.T).T
            test_vector = (self.dico.dictionary @ self.test_image.T).T
        else:
            training_vector = self.shl.code(data=self.training_image, dico=self.dico, matname=self.matname)
            test_vector = self.shl.code(data=self.test_image, dico=self.dico, matname=self.matname)
        #sparse_encode(test_image, dico.dictionary, algorithm = shl.learning_algorithm,
        #                        l0_sparseness=l0_sparseness, fit_tol = None,
        #                        P_cum = dico.P_cum, verbose = 0)
        wrapped_training_data = (training_vector, self.training_supervision)
        wrapped_test_data = (test_vector, self.test_supervision)

        wrapped_inputs = [np.reshape(x, (self.shl.n_dictionary, 1)) for x in wrapped_training_data[0]]
        wrapped_results = [vectorized_result(y) for y in wrapped_training_data[1]]
        wrapped_training_data = zip(wrapped_inputs, wrapped_results)
        wrapped_test_inputs = [np.reshape(x, (self.shl.n_dictionary, 1)) for x in wrapped_test_data[0]]
        wrapped_test_data_final = zip(wrapped_test_inputs, wrapped_test_data[1])
        n_test = len(wrapped_test_inputs)
        return wrapped_training_data, wrapped_test_data_final, n_test

    def learn(self):
        wrapped_training_data, wrapped_test_data_final, n_test = self.format_data()

        self.net.SGD(training_data=wrapped_training_data,
           epochs=self.epochs,
           mini_batch_size=self.mini_batch_size,
           eta=self.eta, test_data=wrapped_test_data_final)
        
    def result(self):
        wrapped_training_data, wrapped_test_data_final, n_test = self.format_data()
        return 1. * self.net.evaluate(wrapped_test_data_final) / n_test