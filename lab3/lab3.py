# MIT 6.034 Lab 3: Constraint Satisfaction Problems
# Written by 6.034 staff

from constraint_api import *
from test_problems import get_pokemon_problem


#### Part 1: Warmup ############################################################

def has_empty_domains(csp) :
    """Returns True if the problem has one or more empty domains, otherwise False"""
    for var in csp.get_all_variables():
        if not csp.get_domain(var):
            return True
    return False

def check_all_constraints(csp) :
    """Return False if the problem's assigned values violate some constraint,
    otherwise True"""
    for constraint in csp.get_all_constraints():
        val1 = csp.get_assignment(constraint.var1)
        val2 = csp.get_assignment(constraint.var2)
        
        if not (val1 is None or val2 is None): 
            if not constraint.check(val1, val2):
                return False
    return True

#### Part 2: Depth-First Constraint Solver #####################################

def solve_constraint_dfs(problem) :
    """
    Solves the problem using depth-first search.  Returns a tuple containing:
    1. the solution (a dictionary mapping variables to assigned values)
    2. the number of extensions made (the number of problems popped off the agenda).
    If no solution was found, return None as the first element of the tuple.
    """
    agenda = [problem]
    solution = None
    num_extensions = 0

    while len(agenda) > 0:
        prob = agenda.pop(0)  
        num_extensions += 1
        if has_empty_domains(prob) or not check_all_constraints(prob):
            continue
        
        if not prob.unassigned_vars:
            solution = prob.assignments
            return (solution, num_extensions)
       
        var = prob.pop_next_unassigned_var()
       
        extensions = []
        for val in prob.get_domain(var):
            new_prob = prob.copy()
            new_prob.set_assignment(var, val)
            extensions.append(new_prob)

        agenda = extensions + agenda

    return (solution, num_extensions)

# QUESTION 1: How many extensions does it take to solve the Pokemon problem
#    with DFS?

# Hint: Use get_pokemon_problem() to get a new copy of the Pokemon problem
#    each time you want to solve it with a different search method.

ANSWER_1 = 20


#### Part 3: Forward Checking ##################################################

def eliminate_from_neighbors(csp, var) :
    """
    Eliminates incompatible values from var's neighbors' domains, modifying
    the original csp.  Returns an alphabetically sorted list of the neighboring
    variables whose domains were reduced, with each variable appearing at most
    once.  If no domains were reduced, returns empty list.
    If a domain is reduced to size 0, quits immediately and returns None.
    """
    reduced = set()
    for neighbor in csp.get_neighbors(var):
        eliminations = []
        for w in csp.get_domain(neighbor):
        
            is_eliminate = True
            for v in csp.get_domain(var):
                checked_constraints = []
                for constraint in csp.constraints_between(var, neighbor):
                    checked_constraints.append(constraint.check(v, w))

                if False not in checked_constraints:
                    is_eliminate = False

            if is_eliminate:
                eliminations.append(w)

        for e in eliminations:
            csp.eliminate(neighbor, e)
            reduced.add(neighbor)

        if not(csp.get_domain(neighbor)):
            return None

    return sorted(reduced)

# Because names give us power over things (you're free to use this alias)
forward_check = eliminate_from_neighbors

def solve_constraint_forward_checking(problem) :
    """
    Solves the problem using depth-first search with forward checking.
    Same return type as solve_constraint_dfs.
    """
    agenda = [problem]
    solution = None
    num_extensions = 0

    while len(agenda) > 0:
        prob = agenda.pop(0)  
        num_extensions += 1
        if has_empty_domains(prob) or not check_all_constraints(prob):
            continue
        
        if not prob.unassigned_vars:
            solution = prob.assignments
            return (solution, num_extensions)
       
        var = prob.pop_next_unassigned_var()
       
        extensions = []
        for val in prob.get_domain(var):
            new_prob = prob.copy()
            new_prob.set_assignment(var, val)
            
            eliminate_from_neighbors(new_prob, var) 

            extensions.append(new_prob)

        agenda = extensions + agenda

    return (solution, num_extensions)


# QUESTION 2: How many extensions does it take to solve the Pokemon problem
#    with DFS and forward checking?

ANSWER_2 = 9


#### Part 4: Domain Reduction ##################################################

def domain_reduction(csp, queue=None) :
    """
    Uses constraints to reduce domains, propagating the domain reduction
    to all neighbors whose domains are reduced during the process.
    If queue is None, initializes propagation queue by adding all variables in
    their default order. 
    Returns a list of all variables that were dequeued, in the order they
    were removed from the queue.  Variables may appear in the list multiple times.
    If a domain is reduced to size 0, quits immediately and returns None.
    This function modifies the original csp.
    """
    if queue is None:
        queue = csp.get_all_variables()
    
    dequeued = []
    while len(queue) > 0:
        var = queue.pop(0)
        reduced = eliminate_from_neighbors(csp, var)
        
        if reduced is None:
            return None

        queue += reduced
        dequeued.append(var)

    return dequeued


# QUESTION 3: How many extensions does it take to solve the Pokemon problem
#    with DFS (no forward checking) if you do domain reduction before solving it?

ANSWER_3 = 6


def solve_constraint_propagate_reduced_domains(problem) :
    """
    Solves the problem using depth-first search with forward checking and
    propagation through all reduced domains.  Same return type as
    solve_constraint_dfs.
    """
    agenda = [problem]
    solution = None
    num_extensions = 0

    while len(agenda) > 0:
        prob = agenda.pop(0)  
        num_extensions += 1
        if not check_all_constraints(prob) or has_empty_domains(prob):
            continue
        
        if not prob.unassigned_vars:
            solution = prob.assignments
            return (solution, num_extensions)
       
        var = prob.pop_next_unassigned_var()
       
        extensions = []
        for val in prob.get_domain(var):
            new_prob = prob.copy()
            new_prob.set_assignment(var, val)
            domain_reduction(new_prob, [var]) 
            extensions.append(new_prob)

        agenda = extensions + agenda

    return (solution, num_extensions)

# QUESTION 4: How many extensions does it take to solve the Pokemon problem
#    with forward checking and propagation through reduced domains?

ANSWER_4 = 7


#### Part 5A: Generic Domain Reduction #########################################

def propagate(enqueue_condition_fn, csp, queue=None) :
    """
    Uses constraints to reduce domains, modifying the original csp.
    Uses enqueue_condition_fn to determine whether to enqueue a variable whose
    domain has been reduced. Same return type as domain_reduction.
    """
    if queue is None:
        queue = csp.get_all_variables()
    
    dequeued = []
    while len(queue) > 0:
        var = queue.pop(0)
        reduced = eliminate_from_neighbors(csp, var)
        
        if reduced is None:
            return None

        for r in reduced:
            if enqueue_condition_fn(csp, r):
                queue.append(r)

        dequeued.append(var)

    return dequeued

def condition_domain_reduction(csp, var) :
    """Returns True if var should be enqueued under the all-reduced-domains
    condition, otherwise False"""
    return True

def condition_singleton(csp, var) :
    """Returns True if var should be enqueued under the singleton-domains
    condition, otherwise False"""
    return len(csp.get_domain(var)) == 1

def condition_forward_checking(csp, var) :
    """Returns True if var should be enqueued under the forward-checking
    condition, otherwise False"""
    return False


#### Part 5B: Generic Constraint Solver ########################################

def solve_constraint_generic(problem, enqueue_condition=None) :
    """
    Solves the problem, calling propagate with the specified enqueue
    condition (a function). If enqueue_condition is None, uses DFS only.
    Same return type as solve_constraint_dfs.
    """
    agenda = [problem]
    solution = None
    num_extensions = 0

    while len(agenda) > 0:
        prob = agenda.pop(0)  
        num_extensions += 1
        if not check_all_constraints(prob) or has_empty_domains(prob):
            continue
        
        if not prob.unassigned_vars:
            solution = prob.assignments
            return (solution, num_extensions)
       
        var = prob.pop_next_unassigned_var()
       
        extensions = []
        for val in prob.get_domain(var):
            new_prob = prob.copy()
            new_prob.set_assignment(var, val)
            if enqueue_condition is not None: 
                propagate(enqueue_condition, new_prob, [var]) 
            extensions.append(new_prob)

        agenda = extensions + agenda

    return (solution, num_extensions)

# QUESTION 5: How many extensions does it take to solve the Pokemon problem
#    with forward checking and propagation through singleton domains? (Don't
#    use domain reduction before solving it.)

ANSWER_5 = 8


#### Part 6: Defining Custom Constraints #######################################

def constraint_adjacent(m, n) :
    """Returns True if m and n are adjacent, otherwise False.
    Assume m and n are ints."""
    return abs(m - n) == 1

def constraint_not_adjacent(m, n) :
    """Returns True if m and n are NOT adjacent, otherwise False.
    Assume m and n are ints."""
    return abs(m - n) != 1

def all_different(variables) :
    """Returns a list of constraints, with one difference constraint between
    each pair of variables."""
    pairs = [(variables[i], variables[j]) for i in range(len(variables)) for j in range(i+1, len(variables))]

    return [Constraint(var1, var2, constraint_different) for var1, var2 in pairs]

#### SURVEY ####################################################################

NAME = "Peyton Wang"
COLLABORATORS = "Marisa Papagelis"
HOW_MANY_HOURS_THIS_LAB_TOOK = 10
WHAT_I_FOUND_INTERESTING = "The step-by-step process of adding on the enhancements"
WHAT_I_FOUND_BORING = "Code felt very repetitive"
SUGGESTIONS = None
