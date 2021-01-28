# MIT 6.034 Lab 9: Boosting (Adaboost)
# Written by 6.034 staff

from math import log as ln
from utils import *


#### Part 1: Helper functions ##################################################

def initialize_weights(training_points):
    """Assigns every training point a weight equal to 1/N, where N is the number
    of training points.  Returns a dictionary mapping points to weights."""
    N = len(training_points)
    mappings = {}

    for point in training_points:
        mappings[point] = make_fraction(1/N)

    return mappings

def calculate_error_rates(point_to_weight, classifier_to_misclassified):
    """Given a dictionary mapping training points to their weights, and another
    dictionary mapping classifiers to the training points they misclassify,
    returns a dictionary mapping classifiers to their error rates."""
    mappings = {}

    for classifier, points in classifier_to_misclassified.items():
        mappings[classifier] = sum([point_to_weight[point] for point in points])

    return mappings

def pick_best_classifier(classifier_to_error_rate, use_smallest_error=True):
    """Given a dictionary mapping classifiers to their error rates, returns the
    best* classifier, or raises NoGoodClassifiersError if best* classifier has
    error rate 1/2.  best* means 'smallest error rate' if use_smallest_error
    is True, otherwise 'error rate furthest from 1/2'."""
    best_classifier = None
    
    if use_smallest_error: 
        smallest_error = 1
    else:
        smallest_error = 1/2 

    for classifier, error in classifier_to_error_rate.items():
        if use_smallest_error: 
            if make_fraction(error) < make_fraction(smallest_error):
                best_classifier = classifier
                smallest_error = make_fraction(error) 
        else: 
            if make_fraction(abs(1/2 - smallest_error)) < make_fraction(abs(1/2 - error)):
                best_classifier = classifier
                smallest_error = make_fraction(error)

    if make_fraction(smallest_error) == 1/2:
        raise NoGoodClassifiersError

    return best_classifier


def calculate_voting_power(error_rate):
    """Given a classifier's error rate (a number), returns the voting power
    (aka alpha, or coefficient) for that classifier."""
    if error_rate == 0:
        return INF
    elif error_rate == 1:
        return -INF
    return 0.5 * ln((1-error_rate)/error_rate)

def get_overall_misclassifications(H, training_points, classifier_to_misclassified):
    """Given an overall classifier H, a list of all training points, and a
    dictionary mapping classifiers to the training points they misclassify,
    returns a set containing the training points that H misclassifies.
    H is represented as a list of (classifier, voting_power) tuples."""
    misclassified = set()

    for point in training_points:
        classification = 0
        for classifier, voting_power in H:
            if point in classifier_to_misclassified[classifier]:
                classification -= voting_power
            else:
                classification += voting_power
        
        if classification <= 0:
            misclassified.add(point)
    
    return misclassified    


def is_good_enough(H, training_points, classifier_to_misclassified, mistake_tolerance=0):
    """Given an overall classifier H, a list of all training points, a
    dictionary mapping classifiers to the training points they misclassify, and
    a mistake tolerance (the maximum number of allowed misclassifications),
    returns False if H misclassifies more points than the tolerance allows,
    otherwise True.  H is represented as a list of (classifier, voting_power)
    tuples."""
    misclassified = get_overall_misclassifications(H, training_points, classifier_to_misclassified)

    return len(misclassified) <= mistake_tolerance

def update_weights(point_to_weight, misclassified_points, error_rate):
    """Given a dictionary mapping training points to their old weights, a list
    of training points misclassified by the current weak classifier, and the
    error rate of the current weak classifier, returns a dictionary mapping
    training points to their new weights.  This function is allowed (but not
    required) to modify the input dictionary point_to_weight."""
    for point in point_to_weight:
        if point in misclassified_points:
            point_to_weight[point] *= make_fraction(0.5 * 1/error_rate)
        else:
            point_to_weight[point] *= make_fraction(0.5 * 1/(1-error_rate))

    return point_to_weight

#### Part 2: Adaboost ##########################################################

def adaboost(training_points, classifier_to_misclassified,
             use_smallest_error=True, mistake_tolerance=0, max_rounds=INF):
    """Performs the Adaboost algorithm for up to max_rounds rounds.
    Returns the resulting overall classifier H, represented as a list of
    (classifier, voting_power) tuples."""
    point_to_weight = initialize_weights(training_points)
    H = []
    
    rounds = 0

    while rounds < max_rounds:
        classifier_to_error_rate = calculate_error_rates(point_to_weight, classifier_to_misclassified)

        if is_good_enough(H, training_points, classifier_to_misclassified, mistake_tolerance):
            break
        try:
            best_classifier = pick_best_classifier(classifier_to_error_rate, use_smallest_error)
        except NoGoodClassifiersError:
            break
        
        error_rate = classifier_to_error_rate[best_classifier]
        point_to_weight = update_weights(point_to_weight, classifier_to_misclassified[best_classifier], error_rate)
        voting_power = calculate_voting_power(error_rate)

        H.append((best_classifier, voting_power))
        rounds += 1
        
    return H


#### SURVEY ####################################################################

NAME = 'Peyton Wang'
COLLABORATORS = 'Marisa Papagelis'
HOW_MANY_HOURS_THIS_LAB_TOOK = 10
WHAT_I_FOUND_INTERESTING = 'Adaboost'
WHAT_I_FOUND_BORING = 'Helper functions'
SUGGESTIONS = 'N/A'
