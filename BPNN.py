'''
A Framework of Back Propagation Neural Network

Just for the purpose of demostration

Author: Zihao Chen
Date: 2019-6-24

'''
import numpy as np
import matplotlib.pyplot as plt 
#from math import exp


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def costFunc(input_X, output_y, label_y, capital_theta, reg_lambda):
    pass
    #Better to reconstruct the training method.


class Layer():
    # a layer of BP Neural Network
    def __init__(self, numofunits, input_label=False, output_label=False):
        self.numofunits = numofunits
        self.input_label = input_label
        self.output_label = output_label
        self.value = np.asmatrix(np.zeros(numofunits)).T
        self.delta = np.asmatrix(np.zeros(numofunits)).T


class BPNN_model():
    # a BPNN model
    def __init__(self, input_size=0, output_size=0, learning_rate=0.1, reg_lambda=1):
        self.input_size = input_size
        self.output_size = output_size
        self.layers = []
        input_layer = Layer(self.input_size, input_label=True)
        self.layers.append(input_layer)
        self.all_Theta = []


    def addlayer(self, layer):
        self.layers.append(layer)


    def save_model(self, all_Theta):
        self.all_Theta = all_Theta

    #Input a test set and generate an output
    def inference(self, test_set):
        m, n = np.shape(test_set)
        num_of_layers = len(self.layers)
        if n != self.input_size:
            print("Error: The size of the testset doesn't match the training set.")
        elif self.all_Theta == []:
            print("Please train the model before do the inference!")
        else:
            input_x = np.asmatrix(test_set)
            act_mat = input_x
            for l in range(num_of_layers-1):
                z = np.hstack((np.matrix(np.ones(m)).T, act_mat)) * self.all_Theta[l].T
                act_mat = sigmoid(z)
            return act_mat > 0.5 

    #!!! output layer will be automatically added
    def train(self, training_set, max_iter=10000, learning_rate=0.1, reg_lambda=1, reset_output=False):
        #first we initialize the theta
        output_size = self.output_size
        if reset_output == True:
            output_layer = Layer(output_size, output_label=True)
            self.layers.append(output_layer)
        num_of_layers = len(self.layers)
        label_count = 0
        for l in range(num_of_layers):
            if self.layers[l].output_label == True:
                label_count += 1
        if label_count != 1:
            print("More than one output layer. You forgot to delete the output layer before adding new hidden layer. Or you didn't set the 'reset_output' to be True")
        elif self.layers[-1].output_label == False:
            print("If you have changed number of layers, please set the parameter 'reset_output' to be True.")
        else: 
            m, n = np.shape(training_set)
            n -= output_size

            #initialize Theta
            capital_theta = []
            for i, layer in enumerate(self.layers):
            # There must be an input layer to be layers[0]
                if i == 0:
                    pass
                else:
                    n_row = layer.numofunits
                    n_col = self.layers[i-1].numofunits + 1
                    theta = np.random.randn(n_row, n_col)
                    theta = np.asmatrix(theta)
                    capital_theta.append(theta)

            #store costFunction in every iteration
            listofJ = []
            #Start training. 
            iteration = 0
            while iteration < max_iter :
                iteration += 1
                costFunc = 0
                #create a list to store DELTA. 
                capital_delta = []
                for theta in capital_theta:
                    delta_l = np.zeros_like(theta)
                    delta_l = np.asmatrix(delta_l)
                    capital_delta.append(delta_l)
                for i in range(m):
                    # For a single sample
                    input_x = np.asmatrix(training_set[i, :n])
                    input_x = input_x.T             
                    lable_y = np.asmatrix(training_set[i, n:])
                    lable_y = lable_y.T
                    self.layers[0].value = input_x
                    #Feedforward propagation
                    for l in range(len(capital_theta)):
                        self.layers[l].value = np.vstack((np.matrix([[1]]), self.layers[l].value))
                        #matrix multiplication
                        self.layers[l+1].value = capital_theta[l] * self.layers[l].value
                        #activate
                        self.layers[l+1].value = sigmoid(self.layers[l+1].value)
                    output_y = self.layers[num_of_layers-1].value
                    #Get the cost           
                    costFunc += - lable_y.T * np.log(output_y) - \
                                (1 - lable_y).T * np.log(1 - output_y)

                    #Backward propagation
                    self.layers[num_of_layers-1].delta = output_y - lable_y
                    for l in np.arange(num_of_layers-2, 0, -1):
                        self.layers[l].delta = capital_theta[l].T * self.layers[l+1].delta
                        self.layers[l].delta = np.multiply(self.layers[l].delta, 
                                            np.multiply(self.layers[l].value, (1 - self.layers[l].value)))
                    for l in np.arange(num_of_layers-1):
                        if l == num_of_layers - 2:
                            capital_delta[l] += self.layers[l+1].delta * self.layers[l].value.T
                        else:
                            capital_delta[l] += self.layers[l+1].delta[1:, :] * self.layers[l].value.T
                #Add regularization term
                sumTheta = 0
                for theta in capital_theta:
                    sumTheta += np.sum(np.multiply(theta, theta))
                costFunc += reg_lambda * sumTheta / 2
                costFunc = costFunc / m
                costFunc = np.asscalar(costFunc)
                print("costFunction: ", costFunc)
                listofJ.append(costFunc)
                #Calculate the gradient
                grad_D = []
                for l in range(num_of_layers-1):
                    grad_l = capital_delta[l] / m
                    grad_l[:, 1:] += reg_lambda * capital_theta[l][:, 1:] / m
                    grad_D.append(grad_l)
                #gradient descending
                    capital_theta[l] = capital_theta[l] - learning_rate * grad_l
        self.save_model(capital_theta)
        listofJ = np.squeeze(listofJ)#no need to squeeze any more
        plt.plot(range(max_iter), listofJ)

# Give an example on how to use the Framework
def example():
    #Input dataset is a set of 10*10 pixel image.  And the output y equals 0 or 1.
    m = 200
    data = np.asmatrix(np.random.randn(m, 100))
    label = np.asmatrix(np.random.randint(2, size=m))
    dataset = np.hstack((data, label.T))
    #construct the model
    mymodel = BPNN_model(input_size=100, output_size=1)
    hiddenlayer = Layer(25)
    mymodel.addlayer(hiddenlayer)
    mymodel.train(dataset, max_iter=1000, learning_rate=1, reset_output=True)
    dataset = dataset[:, :100]
    y = mymodel.inference(dataset)
    accuracy = (y == label.T)
    print("Train accuracy: ", accuracy.mean())
    #test the model
    test_set = np.random.randn(10, 100)
    y = mymodel.inference(test_set)
    print(y)

if __name__ == '__main__':
    example()








            




