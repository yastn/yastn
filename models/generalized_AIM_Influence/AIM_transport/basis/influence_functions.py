import numpy as np


def influence_LinLin(w, W, u):
    """
    %InfluenceLin gives a constant influence, which gives evenly spaced modes
    %   This is a normalized, constant influence of 1. It accepts an array of
    %   frequencies w and returns a same size array of the influence.
    """
    return 1.


def influence_LinInv(w, W, u):
    """
    %InfluenceLinLog gives a lin-inv influence
    %   This is a normalized, constant influence of 1 within the bias window
    %   and a 1/w^2 decay outside the bias window. It accepts an array of
    %   frequencies w and returns a same size array of the influence.
    """
    return np.heaviside(u/2+w, .5)*np.heaviside(u/2-w, .5)+(np.heaviside(w-u/2, .5)+np.heaviside(-u/2-w, .5))*(u/2)**2/(w+np.spacing(1)*abs(abs(np.sign(w))-1))**2


def influence_LinInv1D(w, W, u):
    """
    %InfluenceLinLog gives a lin-inv influence
    %   This is a normalized, constant influence of 1 within the bias window
    %   and a 1/w^2 decay outside the bias window. It accepts an array of
    %   frequencies w and returns a same size array of the influence.
    """
    return np.sqrt(1-(2*w/W)**2)*(np.heaviside(u/2+w, .5)*np.heaviside(u/2-w, .5)+(np.heaviside(w-u/2, .5)+np.heaviside(-u/2-w, .5))*(u/2)**2/(w+np.spacing(1)*abs(abs(np.sign(w))-1))**2)


def influence_LinLog(w, W, u):
    """
    %InfluenceLinLog gives a lin-log influence
    %   This is a normalized, constant influence of 1 within the bias window 
    %   and a 1/w decay outside the bias window. It accepts an array of
    %   frequencies w and returns a same size array of the influence. 
    """
    return np.heaviside(u/2+w, .5)*np.heaviside(u/2-w, .5)+(np.heaviside(w-u/2, .5)+np.heaviside(-u/2-w, .5))*(u/2)/abs(w+np.spacing(1)*abs(abs(np.sign(w))-1))


def influence_LinLog1D(w, W, u):
    """
    %InfluenceLinLog gives a lin-log influence
    %   This is a normalized, constant influence of 1 within the bias window 
    %   and a 1/w decay outside the bias window. It accepts an array of
    %   frequencies w and returns a same size array of the influence. 
    """
    return np.sqrt(1.-(2.*w/W)**2)*np.heaviside(u/2+w, .5)*np.heaviside(u/2-w, .5)+(np.heaviside(w-u/2, .5)+np.heaviside(-u/2-w, .5))*(u/2)/abs(w+np.spacing(1)*abs(abs(np.sign(w))-1))
