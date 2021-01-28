# MIT 6.034 Lab 7: Neural Nets
# Written by 6.034 Staff

from nn_problems import *
from math import e
INF = float('inf')


#### Part 1: Wiring a Neural Net ###############################################

nn_half = [1]

nn_angle = [2, 1]

nn_cross = [2, 2, 1]

nn_stripe = [3, 1]

nn_hexagon = [6, 1]

nn_grid = [4, 2, 1]


#### Part 2: Coding Warmup #####################################################

# Threshold functions
def stairstep(x, threshold=0):
    "Computes stairstep(x) using the given threshold (T)"
    if x >= threshold:
        return 1
    else: 
        return 0

def sigmoid(x, steepness=1, midpoint=0):
    "Computes sigmoid(x) using the given steepness (S) and midpoint (M)"
    return 1 / (1 + e**(-1*(steepness*(x-midpoint))))

def ReLU(x):
    "Computes the threshold of an input using a rectified linear unit."
    return max(0, x)

# Accuracy function
def accuracy(desired_output, actual_output):
    "Computes accuracy. If output is binary, accuracy ranges from -0.5 to 0."
    return -0.5 * (desired_output-actual_output)**2


#### Part 3: Forward Propagation ###############################################

def node_value(node, input_values, neuron_outputs):  # PROVIDED BY THE STAFF
    """
    Given 
     * a node (as an input or as a neuron),
     * a dictionary mapping input names to their values, and
     * a dictionary mapping neuron names to their outputs
    returns the output value of the node.
    This function does NOT do any computation; it simply looks up
    values in the provided dictionaries.
    """
    if isinstance(node, str):
        # A string node (either an input or a neuron)
        if node in input_values:
            return input_values[node]
        if node in neuron_outputs:
            return neuron_outputs[node]
        raise KeyError("Node '{}' not found in either the input values or neuron outputs dictionary.".format(node))
    
    if isinstance(node, (int, float)):
        # A constant input, such as -1
        return node
    
    raise TypeError("Node argument is {}; should be either a string or a number.".format(node))

def forward_prop(net, input_values, threshold_fn=stairstep):
    """Given a neural net and dictionary of input values, performs forward
    propagation with the given threshold function to compute binary output.
    This function should not modify the input net.  Returns a tuple containing:
    (1) the final output of the neural net
    (2) a dictionary mapping neurons to their immediate outputs"""
    mappings = {}
    sorted_net = net.topological_sort()

    for neuron in sorted_net:
        outputs = 0
        for input_node in net.get_incoming_neighbors(neuron):
            wires = net.get_wires(input_node, neuron)
            output_val = node_value(input_node, input_values, mappings)
            outputs += output_val * wires[0].get_weight()
        
        mappings[neuron] = threshold_fn(outputs)

    return (mappings[net.get_output_neuron()], mappings)


#### Part 4: Backward Propagation ##############################################

def gradient_ascent_step(func, inputs, step_size):
    """Given an unknown function of three variables and a list of three values
    representing the current inputs into the function, increments each variable
    by +/- step_size or 0, with the goal of maximizing the function output.
    After trying all possible variable assignments, returns a tuple containing:
    (1) the maximum function output found, and
    (2) the list of inputs that yielded the highest function output."""
    max_output = -INF
    best_inputs = []
    perturb_list = [0, -step_size, step_size]
    
    for i in perturb_list:
        for j in perturb_list:
            for k in perturb_list:
                output = func(inputs[0]+i, inputs[1]+j, inputs[2]+k)
                if output > max_output:
                    max_output = output
                    best_inputs = [inputs[0]+i, inputs[1]+j, inputs[2]+k]

    return (max_output, best_inputs)

def get_back_prop_dependencies(net, wire):
    """Given a wire in a neural network, returns a set of inputs, neurons, and
    Wires whose outputs/values are required to update this wire's weight."""
    dependencies = {wire.startNode, wire, wire.endNode}
    
    wires = net.get_wires(wire.endNode)
    for w in wires:
        dependencies = dependencies.union(get_back_prop_dependencies(net, w))
            
    return dependencies

def calculate_deltas(net, desired_output, neuron_outputs):
    """Given a neural net and a dictionary of neuron outputs from forward-
    propagation, computes the update coefficient (delta_B) for each
    neuron in the net. Uses the sigmoid function to compute neuron output.
    Returns a dictionary mapping neuron names to update coefficient (the
    delta_B values). """
    mappings = {}
    
    sorted_net = net.topological_sort()
    sorted_net.reverse()

    for neuron in sorted_net:
        if net.is_output_neuron(neuron):
            delta_B = neuron_outputs[neuron] * (1-neuron_outputs[neuron]) * (desired_output-neuron_outputs[neuron])
            mappings[neuron] = delta_B
        else:
            sum = 0
            for ci in net.get_outgoing_neighbors(neuron):
                sum += net.get_wires(neuron, ci)[0].get_weight() * mappings[ci]
            
            delta_B = neuron_outputs[neuron] * (1-neuron_outputs[neuron]) * sum
            mappings[neuron] = delta_B
   
    return mappings

def update_weights(net, input_values, desired_output, neuron_outputs, r=1):
    """Performs a single step of back-propagation.  Computes delta_B values and
    weight updates for entire neural net, then updates all weights.  Uses the
    sigmoid function to compute neuron output.  Returns the modified neural net,
    with the updated weights."""
    mappings = calculate_deltas(net, desired_output, neuron_outputs)

    sorted_net = net.topological_sort()
    
    for neuron in net.inputs:
        for wire in net.get_wires(neuron):
            new_weight = wire.get_weight() + (r * node_value(neuron, input_values, neuron_outputs) * mappings[wire.endNode])
            wire.set_weight(new_weight)
    
    for neuron in sorted_net:
        for wire in net.get_wires(neuron):
            new_weight = wire.get_weight() + (r * node_value(neuron, input_values, neuron_outputs) * mappings[wire.endNode])
            wire.set_weight(new_weight)
    
    return net


def back_prop(net, input_values, desired_output, r=1, minimum_accuracy=-0.001):
    """Updates weights until accuracy surpasses minimum_accuracy.  Uses the
    sigmoid function to compute neuron output.  Returns a tuple containing:
    (1) the modified neural net, with trained weights
    (2) the number of iterations (that is, the number of weight updates)"""
    iterations = 0
    output, mappings = forward_prop(net, input_values, sigmoid)
    
    while accuracy(desired_output, output) < minimum_accuracy:
        net = update_weights(net, input_values, desired_output, mappings, r)
        iterations += 1
        output, mappings = forward_prop(net, input_values, sigmoid)
    
    return (net, iterations)


#### Part 5: Training a Neural Net #############################################

ANSWER_1 = 12
ANSWER_2 = 13
ANSWER_3 = 6
ANSWER_4 = 104
ANSWER_5 = 10

ANSWER_6 = 1
ANSWER_7 = "checkerboard"
ANSWER_8 = ["small", "medium", "large"]
ANSWER_9 = "B"

ANSWER_10 = "D"
ANSWER_11 = ["A", "C"]
ANSWER_12 = ["A", "E"]


#### SURVEY ####################################################################

NAME = "Peyton Wang"
COLLABORATORS = "Marisa Papagelis"
HOW_MANY_HOURS_THIS_LAB_TOOK = 6
WHAT_I_FOUND_INTERESTING = None
WHAT_I_FOUND_BORING = None
SUGGESTIONS = None
