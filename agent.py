import sys
import os
sys.path.append(os.path.dirname('__file__'))

from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn as nn
from networks import FFN




class Agent():

    def __init__(self, config):
        
        # The agent knows the model of the environment
        self.b = config["b"]
        self.sigma = config["sigma"]

        # Running reward function and final reward function
        self.f = config["f"]
        self.g = config["g"]

        # BSDE
        self.Y = FFN(sizes=(config["state_dim"]+1, *config["hidden_sizes_Y"], config["state_dim"])) # +1 is for the time
        self.Y = self.Y.to(config["device"])
        self.Z = FFN(sizes=(config["state_dim"]+1, *config["hidden_sizes_Z"], config["state_dim"]))
        self.Z = self.Z.to(config["device"])
        self.optimizer_BSDE = torch.optim.Adam(list(self.Y.parameters())+list(self.Z.parameters()), 
                lr = config["lr"])
        self.loss_fn = nn.MSELoss()
        
        # policy
        self.alpha = FFN(sizes=(config["state_dim"]+1, *config["hidden_sizes_alpha"], config["state_dim"]))
        self.alpha = self.alpha.to(config["device"])
        self.optimizer_alpha = torch.optim.Adam(self.alpha.parameters(), lr=config["lr"])
        
        # Time discretisation
        self.T = config["T"] # horizon time
        self.n_steps = config["n_steps"] # number of steps in time discretisation
        self.timegrid = torch.linspace(0, self.T, self.n_steps+1)
    

    def Hamiltonian(self, t, x, a, y, z):
        """
        Evaluate Hamiltonian
        """
        H = self.b(t,x,a)*y + self.sigma(t,x,a) * z + self.f(t,x,a)
        return H


    def _get_loss_BSDE(self, X0):
        """Forward backward SDE

        Parameters
        ----------
        X0 : torch.Tensor of size (batch_size, dim)
            Initial value of SDE

        Returns
        -------
        loss : torch.tensor
            loss function given from Euler scheme from BSDE

        """
        loss_BSDE = 0
        
        X_old = X0
        Y_old = self.Y(torch.cat([torch.zeros_like(X0),X0],1))
        
        # Euler scheme of BSDE + calculation of loss 
        # for solving the BSDE with final condition Y_T = \partial_x g(X_T)
        for idx, t in enumerate(self.timegrid[:-1]):
            
            h = self.timegrid[idx+1] - self.timegrid[idx]
            dW = torch.randn_like(X_old) * torch.sqrt(h)

            with torch.no_grad():
                input_nn = torch.cat([torch.ones_like(X_old)*t, X_old],1)
                a = self.alpha(input_nn)
            
            # SDE - Euler step
            X_new = X_old + self.b(t,X_old,a)*h + self.sigma(t,X_old,a)*dW
            
            # BSDE - Update loss
            x = X_old.detach().requires_grad_(True)
            input_nn = torch.cat([torch.ones_like(X_old)*t, X_old],1) 
            Z_t = self.Z(input_nn)
            H = self.Hamiltonian(t,x,a,Y_old, Z_t)
            partial_x_of_H = torch.autograd.grad(H,x,only_inputs=True, grad_outputs=torch.ones_like(H),create_graph=False, retain_graph=False)[0] 
            
            if idx < (len(self.timegrid)-2):
                input_nn = torch.cat([torch.ones_like(X_new)*self.timegrid[idx+1], X_new],1)
                Y_target = self.Y(input_nn)
            else:
                # terminal condition: Y_T = \partial_x g(X_T)
                x = X_new.detach().requires_grad_(True)
                g = self.g(x)
                partial_x_of_g = torch.autograd.grad(g,x,grad_outputs=torch.ones_like(g), only_inputs=True, create_graph=False, retain_graph=False)[0]
                Y_target = partial_x_of_g
            #error_from_step_i = Y_target.detach() - (Y_old - partial_x_of_H*h + Z_t*dW)
            Y_pred = Y_old - partial_x_of_H.detach()*h + Z_t*dW
            error_from_step_i = self.loss_fn(Y_pred,Y_target.detach())

            loss_BSDE += error_from_step_i 

            # update values for next step
            X_old = X_new
            Y_old = Y_target
            
        return loss_BSDE

    
    def solve_BSDE(self,X0):
        self.optimizer_BSDE.zero_grad()
        # solve the SDE (with Euler scheme), and do a forward pass
        # on the BSDE and get the loss
        loss_BSDE = self._get_loss_BSDE(X0)
        loss_BSDE.backward()
        self.optimizer_BSDE.step()
        return loss_BSDE

    def _get_loss_policy(self, X0):
        loss_policy = 0
        X = X0
        for idx, t in enumerate(self.timegrid):
            
            # we update the loss --> We want to maximise the Hamiltonian at each t
            input_nn = torch.cat([torch.ones_like(X)*t, X],1)
            with torch.no_grad():
                Y = self.Y(input_nn)
                Z = self.Y(input_nn)
            a = self.alpha(input_nn)
            H = self.Hamiltonian(t,X,a,Y,Z)
            #partial_a_of_H = torch.autograd.grad(H, a, grad_outputs=torch.ones_like(a), only_inputs=True, create_graph=True, retain_graph=False)[0]
            #loss_policy += torch.mean(partial_a_of_H**2)
            loss_policy += H  # careful here. Maybe we should change the policy to be the (squared) of gradient of H with respect to a, so that we can minimise it down to 0
        
            # SDE step (OJU amb el detach que es molt important!!!)
            if t<self.timegrid[-1]:
                h = self.timegrid[idx+1] - self.timegrid[idx]
                dW = torch.randn_like(X) * torch.sqrt(h)
                X = X + self.b(t,X,a.detach())*h + self.sigma(t,X,a.detach())*dW
        
        return loss_policy.sum()
        
        
    def improve_policy(self, X0, maximize=True):
        """
        Maximise the Hamiltonian
        """
        self.optimizer_alpha.zero_grad()
        loss = self._get_loss_policy(X0)
        if maximize:
            loss = -loss # maximizing the loss is minimizing -loss
        loss.backward()
        self.optimizer_alpha.step()
        return loss



