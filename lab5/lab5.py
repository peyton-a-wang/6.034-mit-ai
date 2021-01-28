# MIT 6.034 Lab 5: Bayesian Inference
# Written by 6.034 staff

from nets import *


#### Part 1: Warm-up; Ancestors, Descendents, and Non-descendents ##############

def get_ancestors(net, var):
    "Return a set containing the ancestors of var"
    ancestors = set()
    parents = net.get_parents(var)
    if not parents:
        return ancestors
    else:
        for parent in parents:
            ancestors.add(parent)
            ancestors.update(get_ancestors(net, parent))
    return ancestors

def get_descendants(net, var):
    "Returns a set containing the descendants of var"
    descendants = set()
    children = net.get_children(var)
    if not children:
        return descendants
    else:
        for child in children:
            descendants.add(child)
            descendants.update(get_descendants(net, child))
    return descendants

def get_nondescendants(net, var):
    "Returns a set containing the non-descendants of var"
    non_descendants = set()
    descendants = get_descendants(net, var)
    descendants.add(var)
    variables = net.get_variables()
    
    for var in variables:
        if var not in descendants:
            non_descendants.add(var)
    return non_descendants


#### Part 2: Computing Probability #############################################

def simplify_givens(net, var, givens):
    """
    If givens include every parent of var and no descendants, returns a
    simplified list of givens, keeping only parents.  Does not modify original
    givens.  Otherwise, if not all parents are given, or if a descendant is
    given, returns original givens.
    """
    parents = net.get_parents(var)
    descendants = get_descendants(net, var)
    if parents.issubset(givens) and not descendants.intersection((givens)):
        parents_dict = {}
        for parent in parents:
            parents_dict[parent] = givens[parent]
        return dict(parents_dict.items() & givens.items())
    return givens
    
def probability_lookup(net, hypothesis, givens=None):
    "Looks up a probability in the Bayes net, or raises LookupError"
    simplified_givens = None
    
    if givens:
        var = list(hypothesis.keys())[0]
        simplified_givens = simplify_givens(net, var, givens)
    
    try:
        return net.get_probability(hypothesis, simplified_givens)
    except:
        raise LookupError

def probability_joint(net, hypothesis):
    "Uses the chain rule to compute a joint probability"
    sorted_net = net.topological_sort()
    sorted_net.reverse()
    givens = hypothesis.copy()
    joint_prob = 1 
    
    for var in sorted_net:
        if var in hypothesis:
            val = givens.pop(var)
            joint_prob *= probability_lookup(net, {var: val}, givens)
    return joint_prob
    
def probability_marginal(net, hypothesis):
    "Computes a marginal probability as a sum of joint probabilities"
    variables = net.get_variables()
    combinations = net.combinations(variables, hypothesis)
    marginal_prob = 0 
    
    for c in combinations:
        marginal_prob += probability_joint(net, c)
    return marginal_prob

def probability_conditional(net, hypothesis, givens=None):
    "Computes a conditional probability as a ratio of marginal probabilities"
    if not givens:
        return probability_marginal(net, hypothesis)
    
    for var in hypothesis:
        if var in givens and hypothesis[var] != givens[var]:
            return 0

    return probability_marginal(net, dict(hypothesis, **givens))/probability_marginal(net, givens)
    
def probability(net, hypothesis, givens=None):
    "Calls previous functions to compute any probability"
    return probability_conditional(net, hypothesis, givens)


#### Part 3: Counting Parameters ###############################################

def number_of_parameters(net):
    """
    Computes the minimum number of parameters required for the Bayes net.
    """
    variables = net.get_variables()
    min_params = 0

    for var in variables: 
        var_domain = net.get_domain(var)
        parents = net.get_parents(var)
        
        if not parents:
            min_params += len(var_domain) - 1
        else:
            val = 1
            for parent in parents:
                parent_domain = net.get_domain(parent)
                val *= len(parent_domain)
            min_params += (len(var_domain) - 1) * val

    return min_params


#### Part 4: Independence ######################################################

def is_independent(net, var1, var2, givens=None):
    """
    Return True if var1, var2 are conditionally independent given givens,
    otherwise False. Uses numerical independence.
    """
    var1_domain = net.get_domain(var1)
    var2_domain = net.get_domain(var2)

    for val1 in var1_domain:
        for val2 in var2_domain:
            prob1 = probability(net, {var1: val1, var2: val2}, givens)
            prob2 = probability(net, {var1: val1}, givens) * probability(net, {var2: val2}, givens) 
            
            if not approx_equal(prob1, prob2):
                return False
            
    return True
    
def is_structurally_independent(net, var1, var2, givens=None):
    """
    Return True if var1, var2 are conditionally independent given givens,
    based on the structure of the Bayes net, otherwise False.
    Uses structural independence only (not numerical independence).
    """
    variables = {var1, var2}
    if givens:
        variables.update(givens.keys())

    ancestors = set()
    for var in variables:
        ancestors.update(get_ancestors(net, var))

    variables.update(ancestors)
    subnet = net.subnet(variables)
    
    for ancestor in ancestors:
        children1 = subnet.get_children(ancestor)
        variables.remove(ancestor)
        for var in variables:
            children2 = subnet.get_children(var)
            if children1.intersection(children2):
                subnet.link(ancestor, var)

    subnet.make_bidirectional()

    if givens:
        for g in givens:
            subnet.remove_variable(g)

    return not subnet.find_path(var1, var2)


#### SURVEY ####################################################################

NAME = "Peyton Wang"
COLLABORATORS = "Marisa Papagelis"
HOW_MANY_HOURS_THIS_LAB_TOOK = 8
WHAT_I_FOUND_INTERESTING = "N/A"
WHAT_I_FOUND_BORING = "Calculating the probabilities"
SUGGESTIONS = "N/A"