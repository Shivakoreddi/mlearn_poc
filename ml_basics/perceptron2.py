import numpy as np
import matplotlib.pyplot as plt

class Value:

    def __init__(self,data,_children=(),_op='',label=''):
        self.data=data
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        out = Value(self.data+other.data,(self,other))
        return out

    def __mul__(self, other):
        out = Value(self.data* other.data, (self, other))
        return out


def forward(a,b,c):
    d = a*b+c
    return d

def loss(d,d_cap):
    return d-d_cap

def backward(d_cap):
    pass

