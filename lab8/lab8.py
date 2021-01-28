# MIT 6.034 Lab 8: Support Vector Machines
# Written by 6.034 staff

from svm_data import *
from functools import reduce


#### Part 1: Vector Math #######################################################

def dot_product(u, v):
    """Computes the dot product of two vectors u and v, each represented 
    as a tuple or list of coordinates. Assume the two vectors are the
    same length."""
    dot_prod = 0
    
    for i, j in zip(u, v):
        dot_prod += i*j
   
    return dot_prod

def norm(v):
    """Computes the norm (length) of a vector v, represented 
    as a tuple or list of coords."""
    dot_prod = dot_product(v, v)

    return dot_prod**(1/2)


#### Part 2: Using the SVM Boundary Equations ##################################

def positiveness(svm, point):
    """Computes the expression (w dot x + b) for the given Point x."""
    return dot_product(svm.w, point.coords) + svm.b

def classify(svm, point):
    """Uses the given SVM to classify a Point. Assume that the point's true
    classification is unknown.
    Returns +1 or -1, or 0 if point is on boundary."""
    expr = positiveness(svm, point)

    if expr > 0:
        return 1
    elif expr < 0:
        return -1
    else:
        return 0


def margin_width(svm):
    """Calculate margin width based on the current boundary."""
    denom = (sum([coord ** 2 for coord in svm.w]))**(1/2)
    
    return 2/denom


def check_gutter_constraint(svm):
    """Returns the set of training points that violate one or both conditions:
        * gutter constraint (positiveness == classification, for support vectors)
        * training points must not be between the gutters
    Assumes that the SVM has support vectors assigned."""
    violations = set()
    
    for point in svm.training_points:
        if point in svm.support_vectors and point.classification != positiveness(svm, point):
            violations.add(point)
        elif -1 < positiveness(svm, point) < 1:
            violations.add(point)

    return violations


#### Part 3: Supportiveness ####################################################

def check_alpha_signs(svm):
    """Returns the set of training points that violate either condition:
        * all non-support-vector training points have alpha = 0
        * all support vectors have alpha > 0
    Assumes that the SVM has support vectors assigned, and that all training
    points have alpha values assigned."""
    violations = set()

    for point in svm.training_points:
        if point not in svm.support_vectors and point.alpha != 0:
            violations.add(point)
        elif point in svm.support_vectors and point.alpha <= 0:
            violations.add(point)

    return violations

def check_alpha_equations(svm):
    """Returns True if both Lagrange-multiplier equations are satisfied,
    otherwise False. Assumes that the SVM has support vectors assigned, and
    that all training points have alpha values assigned."""
    sum_eq4 = 0
    w = [0, 0]
    
    for i, point in enumerate(svm.training_points):
        sum_eq4 += point.classification * point.alpha
        if i == 0:
            w = scalar_mult(point.classification * point.alpha, point.coords)
        else: 
            w = vector_add(scalar_mult(point.classification * point.alpha, point.coords), w)

    return sum_eq4 == 0 and w == svm.w


#### Part 4: Evaluating Accuracy ###############################################

def misclassified_training_points(svm):
    """Returns the set of training points that are classified incorrectly
    using the current decision boundary."""
    incorrect = set()

    for point in svm.training_points:
        if not classify(svm, point) == point.classification:
            incorrect.add(point)

    return incorrect    


#### Part 5: Training an SVM ###################################################

def update_svm_from_alphas(svm):
    """Given an SVM with training data and alpha values, use alpha values to
    update the SVM's support vectors, w, and b. Return the updated SVM."""
    support_vectors = []
    w = [0, 0]
    for point in svm.training_points:
        if point.alpha > 0:
            support_vectors.append(point)
            w = vector_add(scalar_mult(point.classification * point.alpha, point.coords), w)

    b_pos = []
    b_neg = []
    for vector in support_vectors:
        b = vector.classification - dot_product(w, vector.coords)
        if vector.classification < 0:
            b_neg.append(b)
        else: 
            b_pos.append(b)

    b = (min(b_neg)+max(b_pos)) / 2
    svm.support_vectors = support_vectors
    
    return svm.set_boundary(w, b)


#### Part 6: Multiple Choice ###################################################

ANSWER_1 = 11
ANSWER_2 = 6
ANSWER_3 = 3
ANSWER_4 = 2

ANSWER_5 = ['A', 'D']
ANSWER_6 = ['A', 'B', 'D']
ANSWER_7 = ['A', 'B', 'D']
ANSWER_8 = []
ANSWER_9 = ['A', 'B', 'D']
ANSWER_10 = ['A', 'B', 'D']

ANSWER_11 = False
ANSWER_12 = True
ANSWER_13 = False
ANSWER_14 = False
ANSWER_15 = False
ANSWER_16 = True

ANSWER_17 = [1, 3, 6, 8]
ANSWER_18 = [1, 2, 4, 5, 6, 7, 8]
ANSWER_19 = [1, 2, 4, 5, 6, 7, 8]

ANSWER_20 = 6


#### SURVEY ####################################################################

NAME = "Peyton Wang"
COLLABORATORS = "Marisa Papagelis"
HOW_MANY_HOURS_THIS_LAB_TOOK = 10
WHAT_I_FOUND_INTERESTING = "Training the SVM"
WHAT_I_FOUND_BORING = "N/A"
SUGGESTIONS = "N/A"
