# Backpropagation

Problem Statement - Evaluate the effectiveness of replacing sigmoid functions by piecewise linear functions consisting of a fixed number (n>2) of linear segments, each of which has a constant gradient, and only the first and last segments have gradient 0.

• Select a non-trivial function approximation problem from the UCI or Stanford
databases, with d inputs and 1 output.

• Train a (traditional) single hidden layer d-2d-1 feedforward neural network with
sigmoid functions using backpropagation.

• Replace each sigmoid node in the hidden layer by a piecewise linear function
with n=4 linear segments; train the resulting network on the same data.

• Repeat, for other values of n (=6, 8, 10, 12, 14).
