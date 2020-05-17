import sys
import os
sys.path.append(os.path.dirname('__file__'))

import numpy as np
import torch
import torch.nn as nn
import argparse

from agent import Agent

# TODO: create class for the environment/model/SDE to put all the coefficients of the SDE and running reward/final reward in there


def b(t,x,a):
    """Drift of the Controlled SDE
    b(t,x,a) = H(t)x + M(t)a

    """
    def H(t):
        return 1
    def M(t):
        return 1

    return H(t)*x + M(t)*a


def sigma(t,x,a):
    """Diffusion of the controlled SDE
    """
    return 0.3


def f(t,x,a):
    """Running cost of the control problem
    f(t,x,a) = C(t)x^2 + D(t)a^2 with C(t)<= 0, D = D(t)-delta<0 forall t, for some delta>0
    """
    def C(t):
        return -1
    
    def D(t):
        return -1

    return C(t)*x**2 + D(t)*a**2

def g(x):
    """Final reward of the control problem
    g(x) = R * x^2, with R<=0
    """
    R = -1
    return R * x**2


def solve_ricatti_equation(config):
    """
    Euler scheme to solve Ricatti equation (not very elaborated)
    """
    def C(t):
        return -1
    def D(t):
        return -1
    def M(t):
        return 1
    def H(t):
        return 1
    R = -1
    timegrid = np.linspace(0,config["T"],config["n_steps"]+1)
    S = np.zeros_like(timegrid)
    h = timegrid[1]-timegrid[0]
    for idx, t in enumerate(timegrid):
        if t==timegrid[-1]:
            S[idx] = R
        else:
            S[idx+1] = S[idx] - h*(1/D(t)*M(t)**2*S[idx]**2 - 2*H(t)*S[idx] - C(t))
    # we reverse S because we had done a change of variable
    S = np.flip(S)
    return S
            

def train_agent(config):
    
    # Analytical solution
    S = solve_ricatti_equation(config)
    # TODO: implement analytical solution of LQR problem using solution of Ricatti equation
    
    agent = Agent(config)

    for it in range(config["training_iterations"]):
        for it_BSDE in range(config["training_BSDE_iterations"]):
            X0 = torch.ones(config["batch_size"], config["state_dim"], device=config["device"])*config["x"]
            loss_BSDE = agent.solve_BSDE(X0)
            print("Solving BSDE, loss={:.4f}".format(loss_BSDE.item()))
        for it_policy in range(config["training_policy_iterations"]):
            X0 = torch.ones(config["batch_size"], config["state_dim"], device=config["device"])*config["x"]
            loss_policy = agent.improve_policy(X0)
            print("Improving Policy, loss={:.4f}".format(loss_policy.item()))
            

    return agent







if __name__=='__main__':

    parser = argparse.ArgumentParser(description="Pontryagin maximum principle")
    parser.add_argument("--device", type=int, default=0)

    args = parser.parse_args()

    if torch.cuda.is_available():
        device = "cuda:{}".format(args.device)
    else:
        device="cpu"

    CONFIG = {"device":device,
            "batch_size":32,
            "training_iterations":1000,
            "training_BSDE_iterations":50,
            "training_policy_iterations":10,
            "lr":0.001,
            "state_dim":1,
            "b":b,
            "sigma":sigma,
            "f":f,
            "g":g,
            "hidden_sizes_alpha":(16,8),
            "hidden_sizes_Y":(16,8),
            "hidden_sizes_Z":(16,8),
            "state_dim":1,
            "x":1,
            "T":1,
            "n_steps":100}
    
    train_agent(CONFIG)
    torch.save("agent_policy.pth.tar", agent.alpha.state_dict())

