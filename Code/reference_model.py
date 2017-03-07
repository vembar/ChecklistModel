from model import  GRU 
import torch
import torch.nn as nn
from utilities import uniform_init, orthogonal_init

class CheckListModel(object):

    def initialize_params(self):
        self.W_o = uniform_init((self.vocab_dim, self.embedding_dim))
        self.b_o = uniform_init((self.vocab_dim))
        self.o_t = uniform_init((1, 3))
        self.w_t = uniform_init((self.vocab_dim))  
        self.f_t = uniform_init((1, 3)) # f_t_gru, f_t_new, f_t_used

        self.P = uniform_init((self.embedding_dim, self.embedding_dim))

        self.E_t_new = uniform_init((self.checklist_length, self.embedding_dim))
        self.alpha_t_new = uniform_init((self.checklist_length))
        self.c_t_new = uniform_init((self.checklist_length))
        self.a_t_new = uniform_init((self.checklist_length))

        self.E_t_used = uniform_init((self.checklist_length, self.embedding_dim))
        self.alpha_t_used = uniform_init((self.checklist_length))
        self.c_t_used = uniform_init((self.checklist_length))
        self.a_t_used = uniform_init((self.checklist_length))

    def __init__(self, vocab_dim, embedding_dim, checklist_length, gamma):

        # network hyperparameters
        self.gamma = gamma  # temperature hyperparameter
        self.vocab_dim = vocab_dim
        self.checklist_length = checklist_length
        self.embedding_dim = embedding_dim

        # other units
        self.GRU_lang = GRU(self.vocab_dim, self.embedding_dim)

        # weights and biases
        self.W_o = nn.Parameter(torch.Tensor(self.vocab_dim, self.embedding_dim))
        self.b_o = nn.Parameter(torch.Tensor(self.vocab_dim))

        # checklist parameters
        self.P = uniform_init((self.embedding_dim, self.embedding_dim))

        self.E_t_new = nn.Parameter(torch.Tensor(self.checklist_length, self.embedding_dim))
        self.alpha_t_new = nn.Parameter(torch.Tensor(self.checklist_length))
        self.c_t_new = nn.Parameter(torch.Tensor(self.checklist_length))
        self.a_t_new = nn.Parameter(torch.Tensor(self.checklist_length))

        self.E_t_used = nn.Parameter(torch.Tensor(self.checklist_length, self.embedding_dim))
        self.alpha_t_used = nn.Parameter(torch.Tensor(self.checklist_length))
        self.c_t_used = nn.Parameter(torch.Tensor(self.checklist_length))
        self.a_t_used = nn.Parameter(torch.Tensor(self.checklist_length))

        # projection matrices
        self.P = nn.Parameter(torch.Tensor(self.embedding_dim, self.embedding_dim))

        # decoding parameters
        self.W_o = nn.Parameter(torch.Tensor(self.vocab_dim, self.embedding_dim))
        self.b_o = nn.Parameter(torch.Tensor(self.vocab_dim))
        self.o_t = nn.Parameter(torch.Tensor(1, 3))
        self.w_t = nn.Parameter(torch.Tensor(self.vocab_dim))
        self.f_t = nn.Parameter(torch.Tensor(1, 3))  # f_t_gru, f_t_new, f_t_used

    def build_model(self):
        pass
