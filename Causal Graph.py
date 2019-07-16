import torch
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np
import torch.nn.init as init
import torch.utils.data as datautil



class TransferFunction(nn.Module):
""" Defines the Causal Function between Random Variables """
    def __init__(self):
        super(TransferFunction, self).__init__()
        # self.linear = nn.Sequential(nn.Linear(1, 5),
        #                         nn.Linear(5, 2))
        self.linear = nn.Sequential(nn.Linear(1, 5),
                                    nn.Linear(5, 1))

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

class Node():
    """Corresponds to the Random Variables in a graph
        Input is the mean, std dev, as well as parent and child (set to None if not given)
        Optional input for string name
    """

    def __init__(self, mu, std, parent=None, child=None, name = None):
        self.mu = mu
        self.std = std
        self.parent = parent
        self.child = child
        self.name = name


    def add_child(self, child):
        self.child = child
    def add_parent(self, parent):
        self.parent = parent
    def new_param(self, sampledparent= None):
        """ Function used to update current node's mean and std dev
            based on the causal function with input being a sample of the parent
        """
        output = causal_func(sampledparent)
        # self.mu, self.std = output.tolist()[0], abs(output.tolist()[1])
        self.mu = output.tolist()

    # def set_mu(self, new_mu):
    #     self.mu = new_mu
    # def set_std(self, new_std):
    #     self.std= new_std

    def sample(self):
        """ Sample value of Variable Gaussian with current parameters """
        self.sampled_val = np.random.normal(self.mu, self.std)
        return self.sampled_val


class DataGenerator(nn.Module):

    def __init__(self, pointer_var, nodes_list):
        super(DataGenerator, self).__init__()
        self.data_dict = {node.name: [] for node in nodes_list} #Used to store all data generated (output)
        self.nodes_list = nodes_list
        self.have_cuda = True
        self.sample_dict={}  # For each run-through the graph store the sampled values
        self.mu_dict = {node.name: [] for node in nodes_list}
        #Record mu values for B in each sample for comparison in model

    def generate_sample(self, key):
        """ Recursive function that uses samples all ancestors of current node and samples the current one"""
        if key.parent:
            if key.parent not in self.sample_dict.keys():
                self.generate_sample(key.parent)
            key.new_param(self.sample_dict[key.parent])
        sampled_val = key.sample()
        self.sample_dict.update({key:sampled_val})


    def generate_data(self, size):
        """ Output the final dictionary with all samples stored as single-value tensors"""
        for i in range(size):
            self.sample_dict={}
            for key in self.nodes_list:
                if key not in self.sample_dict.keys():
                    self.generate_sample(key)
                self.data_dict[key.name].append(torch.Tensor([self.sample_dict[key]]))
                self.mu_dict[key.name].append(torch.Tensor([key.mu]))
        return self.data_dict, self.mu_dict

if __name__ == "__main__":
    causal_func = TransferFunction()
    #Generate Data for all alpha+mu values of A
    for i in range(-10,11):
        #Generate Variables (Nodes)
        A = Node(i,0.5, name = 'A')
        sample_a = A.sample()
        output = causal_func(torch.tensor([sample_a], dtype=torch.float32))
        # bmu, bstd = output.tolist()[0], abs(output.tolist()[1])
        bmu, bstd = output.tolist()[0], 0.5
        B = Node(bmu, bstd, name = 'B')

        #Generate Data
        G = DataGenerator(A, [A,B])
        if i == 0:
            data, mu = G.generate_data(10000)
        else:
            data, mu = G.generate_data(100)

        #turn dictionary lists into tensors
        stacked_input= torch.stack(data['A'])
        stacked_output = torch.stack(data['B'])
        stacked_mu_B= torch.stack(mu['B'])
        #Save tensors into HDF5 Document
        dataset = datautil.TensorDataset(stacked_input, stacked_output, stacked_mu_B)
        torch.save(dataset, 'Causal data A-'+str(i)+'.hdf5')

    #Test file saving done above by loading dataset
    catchdataset = torch.load('Causal data A-10.hdf5')
    loader = datautil.DataLoader(
        catchdataset,
        batch_size=1
    )
    count=0
    for x, l, mu in loader:
        count += 1
        print("SHAPES")
        print(x, l)
        print("VALUES")
        if count > 20:
            break

    print(data)

