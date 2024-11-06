import torch as th
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.optim.adam import Adam
from torch.distributions import Normal

from typing import Optional

# Intrinsic Curiosity Module
# action loss using negative log likelihood 
class ICM(nn.Module):
    def __init__(self,
                 observation_shape:int=11,
                 action_shape:int=2,
                 activation_func:type[nn.Module]=nn.Tanh,
                 layer_size:int=2,
                 hidden_size:int=256,
                 alpha:float=0.1,
                 beta:float=0.2,
                #  optimizer_class:type[Optimizer]=Adam,
                 verbose:Optional[bool]=False):
        super(ICM, self).__init__()
        self.alpha = alpha
        self.beta = beta
        # forward model takes inputs state/feature and action
        # to predict next state / next_feature 
        forward_model = nn.ModuleList()

        assert layer_size > 1 

        forward_model.append(nn.Linear(observation_shape+action_shape,
                                       hidden_size))
        forward_model.append(activation_func())   
        
        for i in range(layer_size-1):
            if(i!=layer_size-2):
                forward_model.append(nn.Linear(hidden_size, 
                                            hidden_size))
                forward_model.append(activation_func())
            else:
                forward_model.append(nn.Linear(hidden_size, 
                                            observation_shape))

        self.forward_model = nn.Sequential(*forward_model)
        
        if(verbose):
            print(f"Forward Model arch:\n{self.forward_model}")

        # Inverse model takes inputs state/feature and next state/next feature
        # to predict action
        inverse_model = nn.ModuleList()

        assert layer_size > 1 

        inverse_model.append(nn.Linear(observation_shape*2,
                                       hidden_size))
        inverse_model.append(activation_func())   
        
        for i in range(layer_size-2):
            forward_model.append(nn.Linear(hidden_size, 
                                        hidden_size))
            forward_model.append(activation_func())

        self.inverse_model = nn.Sequential(*inverse_model)
        self.inverse_model_mu = nn.Linear(hidden_size, action_shape)
        self.inverse_model_sigma = nn.Linear(hidden_size, action_shape)
        
        if(verbose):
            print(f"Inverse Model arch:\n{self.inverse_model}")
            print(f"Inverse Model mu:\n{self.inverse_model_mu}")
            print(f"Inverse Model sigma:\n{self.inverse_model_sigma}")
        
    def forward(self, state, action, next_state):
        if(type(state)!=th.Tensor):
            state = th.tensor(state, dtype=th.float)
        if(type(action)!=th.Tensor):
            action = th.tensor(action, dtype=th.float)
        if(type(next_state)!=th.Tensor):
            next_state = th.tensor(next_state, dtype=th.float)

        state = th.atleast_2d(state)
        action = th.atleast_2d(action)
        next_state = th.atleast_2d(next_state)
        
        # predict action
        pred_act_feat = self.inverse_model(th.cat([state, next_state], dim=1))
        pred_act_mean = self.inverse_model_mu(pred_act_feat)
        pred_act_sigma = th.exp(self.inverse_model_sigma(pred_act_feat))

        # predict next_state 
        pred_next_state = self.forward_model(th.cat([state, action], dim=1))

        return pred_act_mean, pred_act_sigma, pred_next_state
    
    def calc_loss(self, state, action, next_state):
        if(type(state)!=th.Tensor):
            state = th.tensor(state, dtype=th.float)
        if(type(action)!=th.Tensor):
            action = th.tensor(action, dtype=th.float)
        if(type(next_state)!=th.Tensor):
            next_state = th.tensor(next_state, dtype=th.float)

        state = th.atleast_2d(state)
        action = th.atleast_2d(action)
        next_state = th.atleast_2d(next_state)
        
        pred_act_mean, pred_act_sigma, pred_next_state = self.forward(state, 
                                                                      action,
                                                                      next_state)
        
        # inverse loss calculation
        pred_act_distribution = Normal(loc=pred_act_mean, 
                                       scale=pred_act_sigma)
        # print(f"{action=}")
        # print(f"{pred_act_distribution.loc=}")
        # print(f"{pred_act_distribution.scale=}")
        pred_act_log_prob = pred_act_distribution.log_prob(action).sum(dim=-1)
        # print(f"{pred_act_log_prob=}")
        inverse_loss = -pred_act_log_prob.mean()*(1-self.beta)
        # print(f"{inverse_loss=}")

        #forward loss calculation
        fw_loss = nn.MSELoss()
        forward_loss = self.beta*fw_loss(pred_next_state, next_state)

        intrinsic_reward = self.alpha*((pred_next_state - next_state).pow(2)).mean(dim=1)

        return intrinsic_reward, inverse_loss, forward_loss




