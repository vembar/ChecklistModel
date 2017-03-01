import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F
from utilities import uniform_init, orthogonal_init


class GRU(nn.Module):

    def initialize_params(self):
        self.w_z = uniform_init((self.input_dim, self.hidden_dim))
        self.u_z = orthogonal_init((self.hidden_dim, self.hidden_dim))
        self.b_z = uniform_init(self.hidden_dim)

        self.w_r = uniform_init((self.input_dim, self.hidden_dim))
        self.u_r = orthogonal_init((self.hidden_dim, self.hidden_dim))
        self.b_r = uniform_init(self.hidden_dim)

        self.w_c = uniform_init((self.input_dim, self.hidden_dim))
        self.u_c = orthogonal_init((self.hidden_dim, self.hidden_dim))
        self.b_c = uniform_init(self.hidden_dim)

        self.z = uniform_init((self.hidden_dim, self.hidden_dim))
        self.y = uniform_init((self.hidden_dim, self.hidden_dim))
        self.w_s = uniform_init((self.hidden_dim, self.hidden_dim))
        self.u_s = uniform_init((self.hidden_dim, self.hidden_dim))
        self.w_q = uniform_init((self.hidden_dim, self.hidden_dim))
        self.u_q = uniform_init((self.hidden_dim, self.hidden_dim))
        self.u_g = uniform_init((self.hidden_dim, self.hidden_dim))

    def __init__(self, input_dim, hidden_dim, with_batch=False, name='GRU'):
        """
        Initialize neural network.
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.with_batch = with_batch
        self.name = name

        # gate weights and biases
        self.w_z = nn.Parameter(torch.Tensor(self.input_dim, self.hidden_dim))
        self.u_z = nn.Parameter(torch.Tensor(self.hidden_dim, self.hidden_dim))
        self.b_z = nn.Parameter(torch.Tensor(self.hidden_dim))

        # reset gate weights and biases
        self.w_r = nn.Parameter(torch.Tensor(self.input_dim, self.hidden_dim))
        self.u_r = nn.Parameter(torch.Tensor(self.hidden_dim, self.hidden_dim))
        self.b_r = nn.Parameter(torch.Tensor(self.hidden_dim))

        # new memory content weights
        self.w_c = nn.Parameter(torch.Tensor(self.input_dim, self.hidden_dim))
        self.u_c = nn.Parameter(torch.Tensor(self.hidden_dim, self.hidden_dim))
        self.b_c = nn.Parameter(torch.Tensor(self.hidden_dim))

        # self.h_0 = nn.Parameter(torch.Tensor(self.hidden_dim))

        # changes in GRU unit to incorporate checklists
        self.z = nn.Parameter(torch.Tensor(self.hidden_dim, self.hidden_dim))
        self.y = nn.Parameter(torch.Tensor(self.hidden_dim, self.hidden_dim))
        self.w_s = nn.Parameter(torch.Tensor(self.hidden_dim, self.hidden_dim))
        self.w_q = nn.Parameter(torch.Tensor(self.hidden_dim, self.hidden_dim))
        self.u_s = nn.Parameter(torch.Tensor(self.hidden_dim, self.hidden_dim))
        self.u_q = nn.Parameter(torch.Tensor(self.hidden_dim, self.hidden_dim))
        self.u_g = nn.Parameter(torch.Tensor(self.hidden_dim, self.hidden_dim))

    def forward(self, input_sample, g, et_new):
        self.initialize_params()

        def recurrence(x_t, h_t, g, et_new):
            z_t = F.sigmoid(F.linear(x_t, self.w_z) + F.linear(h_t, self.u_z) + self.b_z)
            r_t = F.sigmoid(F.linear(x_t, self.w_r) + F.linear(h_t, self.u_r) + self.b_r)
            s_t = F.sigmoid(F.linear(x_t, self.w_s) + F.linear(h_t, self.u_s))
            q_t = F.sigmoid(F.linear(x_t, self.w_q) + F.linear(h_t, self.u_q))
            c_t = F.tanh(F.linear(x_t, self.w_c) + torch.mul(r_t, F.linear(h_t, self.u_c)) +
                         torch.mul(s_t, F.linear(g, self.y)) + torch.mul(q_t,
                         F.linear(et_new.sum(1).t(), F.linear(self.etnew, self.z))) + self.b_c)
            h_t1 = (1 - z_t) * h_t + z_t * c_t
            return h_t1, c_t

        hidden = torch.randn(self.hidden_dim, 1)
        hidden = F.linear(h_0, self.u_g)

        output = []
        steps = range(input_sample.size(0))
        for i in steps:
            hidden = recurrence(input_sample[i], g, et_new)
            output.append(isinstance(hidden, tuple) and hidden[0] or hidden)

        output = torch.cat(output, 0).view(input_sample.size(0), *output[0].size())

        return output, hidden

