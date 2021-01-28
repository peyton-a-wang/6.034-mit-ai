# MIT 6.034 Lab 6: k-Nearest Neighbors and Identification Trees
# Written by 6.034 Staff

from api import *
from data import *
import math
log2 = lambda x: math.log(x, 2)
INF = float('inf')


################################################################################
############################# IDENTIFICATION TREES #############################
################################################################################


#### Part 1A: Classifying points ###############################################

def id_tree_classify_point(point, id_tree):
    """Uses the input ID tree (an IdentificationTreeNode) to classify the point.
    Returns the point's classification."""
    if id_tree.is_leaf():
        return id_tree.get_node_classification()
    else:
        return id_tree_classify_point(point, id_tree.apply_classifier(point))


#### Part 1B: Splitting data with a classifier #################################

def split_on_classifier(data, classifier):
    """Given a set of data (as a list of points) and a Classifier object, uses
    the classifier to partition the data.  Returns a dict mapping each feature
    values to a list of points that have that value."""
    mappings = {}
    for point in data:
        c = classifier.classify(point)
        if c in mappings:
            mappings[c].append(point)
        else:
            mappings[c] = [point] 
        mappings[classifier.classify(point)] 

    return mappings


#### Part 1C: Calculating disorder #############################################

def branch_disorder(data, target_classifier):
    """Given a list of points representing a single branch and a Classifier
    for determining the true classification of each point, computes and returns
    the disorder of the branch."""
    disorder = 0
    mappings = split_on_classifier(data, target_classifier)
    
    for classifier, classification in mappings.items():
        nb = len(classification)
        nbc = len(data)
        disorder += (-nb/nbc) * log2(nb/nbc)
    
    return disorder

def average_test_disorder(data, test_classifier, target_classifier):
    """Given a list of points, a feature-test Classifier, and a Classifier
    for determining the true classification of each point, computes and returns
    the disorder of the feature-test stump."""
    disorder = 0
    mappings = split_on_classifier(data, test_classifier)   
   
    for classifier, classification in mappings.items():
        nb = len(classification)
        nbc = len(data)
        b_disorder = branch_disorder(classification, target_classifier)
        weight = nb/nbc
        disorder += b_disorder * weight
    
    return disorder


## To use your functions to solve part A2 of the "Identification of Trees"
## problem from 2014 Q2, uncomment the lines below and run lab6.py:

# for classifier in tree_classifiers:
#     print(classifier.name, average_test_disorder(tree_data, classifier, feature_test("tree_type")))


#### Part 1D: Constructing an ID tree ##########################################

def find_best_classifier(data, possible_classifiers, target_classifier):
    """Given a list of points, a list of possible Classifiers to use as tests,
    and a Classifier for determining the true classification of each point,
    finds and returns the classifier with the lowest disorder.  Breaks ties by
    preferring classifiers that appear earlier in the list.  If the best
    classifier has only one branch, raises NoGoodClassifiersError."""
    disorder = INF
    best_classifier = possible_classifiers[0]

    for classifier in possible_classifiers:
        avg_disorder = average_test_disorder(data, classifier, target_classifier)
        if avg_disorder < disorder:
            disorder = avg_disorder
            best_classifier = classifier

    split_branches = len(split_on_classifier(data, best_classifier))
    
    if split_branches == 1:
        raise NoGoodClassifiersError
    
    return best_classifier


## To find the best classifier from 2014 Q2, Part A, uncomment:
# print(find_best_classifier(tree_data, tree_classifiers, feature_test("tree_type")))

def construct_greedy_id_tree(data, possible_classifiers, target_classifier, id_tree_node=None):
    """Given a list of points, a list of possible Classifiers to use as tests,
    a Classifier for determining the true classification of each point, and
    optionally a partially completed ID tree, returns a completed ID tree by
    adding classifiers and classifications until either perfect classification
    has been achieved, or there are no good classifiers left."""
    if not id_tree_node:
        id_tree_node = IdentificationTreeNode(target_classifier)

    if branch_disorder(data, target_classifier) == 0:
        id_tree_node.set_node_classification(target_classifier.classify(data[0]))
    else:
        try:
            best_classifier = find_best_classifier(data, possible_classifiers, target_classifier)
        except NoGoodClassifiersError:
            return id_tree_node

        features = split_on_classifier(data, best_classifier)
        id_tree_node.set_classifier_and_expand(best_classifier, features)
        
        branches = id_tree_node.get_branches()
        for branch_name, child_node in branches.items():
            construct_greedy_id_tree(features[branch_name], possible_classifiers, target_classifier, child_node) 

    return id_tree_node


## To construct an ID tree for 2014 Q2, Part A:
# print(construct_greedy_id_tree(tree_data, tree_classifiers, feature_test("tree_type")))

## To use your ID tree to identify a mystery tree (2014 Q2, Part A4):
# tree_tree = construct_greedy_id_tree(tree_data, tree_classifiers, feature_test("tree_type"))
# print(id_tree_classify_point(tree_test_point, tree_tree))

## To construct an ID tree for 2012 Q2 (Angels) or 2013 Q3 (numeric ID trees):
# print(construct_greedy_id_tree(angel_data, angel_classifiers, feature_test("Classification")))
# print(construct_greedy_id_tree(numeric_data, numeric_classifiers, feature_test("class")))


#### Part 1E: Multiple choice ##################################################

ANSWER_1 = "bark_texture"
ANSWER_2 = "leaf_shape"
ANSWER_3 = "orange_foliage"

ANSWER_4 = [2, 3]
ANSWER_5 = [3]
ANSWER_6 = [2]
ANSWER_7 = 2

ANSWER_8 = "No"
ANSWER_9 = "No"


#### OPTIONAL: Construct an ID tree with medical data ##########################

## Set this to True if you'd like to do this part of the lab
DO_OPTIONAL_SECTION = False

if DO_OPTIONAL_SECTION:
    from parse import *
    medical_id_tree = construct_greedy_id_tree(heart_training_data, heart_classifiers, heart_target_classifier_discrete)


################################################################################
############################# k-NEAREST NEIGHBORS ##############################
################################################################################

#### Part 2A: Drawing Boundaries ###############################################

BOUNDARY_ANS_1 = 3
BOUNDARY_ANS_2 = 4

BOUNDARY_ANS_3 = 1
BOUNDARY_ANS_4 = 2

BOUNDARY_ANS_5 = 2
BOUNDARY_ANS_6 = 4
BOUNDARY_ANS_7 = 1
BOUNDARY_ANS_8 = 4
BOUNDARY_ANS_9 = 4

BOUNDARY_ANS_10 = 4
BOUNDARY_ANS_11 = 2
BOUNDARY_ANS_12 = 1
BOUNDARY_ANS_13 = 4
BOUNDARY_ANS_14 = 4


#### Part 2B: Distance metrics #################################################

def dot_product(u, v):
    """Computes dot product of two vectors u and v, each represented as a tuple
    or list of coordinates.  Assume the two vectors are the same length."""
    dot_prod = 0
    
    for i, j in zip(u, v):
        dot_prod += i*j
   
    return dot_prod

def norm(v):
    "Computes length of a vector v, represented as a tuple or list of coords."
    dot_prod = dot_product(v, v)

    return math.sqrt(dot_prod)

def euclidean_distance(point1, point2):
    "Given two Points, computes and returns the Euclidean distance between them."
    dist = 0
    
    for ui, vi in zip(point1.coords, point2.coords):
        dist += (ui-vi)**2
    
    return math.sqrt(dist)

def manhattan_distance(point1, point2):
    "Given two Points, computes and returns the Manhattan distance between them."
    dist = 0
    
    for ui, vi in zip(point1.coords, point2.coords):
        dist += abs(ui-vi)
    
    return dist

def hamming_distance(point1, point2):
    "Given two Points, computes and returns the Hamming distance between them."
    dist = 0
    
    for ui, vi in zip(point1.coords, point2.coords):
        if ui != vi:
            dist += 1

    return dist

def cosine_distance(point1, point2):
    """Given two Points, computes and returns the cosine distance between them,
    where cosine distance is defined as 1-cos(angle_between(point1, point2))."""
    u = point1.coords
    v = point2.coords
    
    dot_prod = dot_product(u, v)
    denom = norm(u) * norm(v)

    return 1 - dot_prod/denom

#### Part 2C: Classifying points ###############################################

def get_k_closest_points(point, data, k, distance_metric):
    """Given a test point, a list of points (the data), an int 0 < k <= len(data),
    and a distance metric (a function), returns a list containing the k points
    from the data that are closest to the test point, according to the distance
    metric.  Breaks ties lexicographically by coordinates."""
    sort_lexo = sorted(data, key=lambda p: p.coords)
    sort_dist = sorted(sort_lexo, key=lambda p: distance_metric(p, point))
    
    return sort_dist[:k]

def knn_classify_point(point, data, k, distance_metric):
    """Given a test point, a list of points (the data), an int 0 < k <= len(data),
    and a distance metric (a function), returns the classification of the test
    point based on its k nearest neighbors, as determined by the distance metric.
    Assumes there are no ties."""
    k_closest = [point.classification for point in get_k_closest_points(point, data, k, distance_metric)]
    
    return max(set(k_closest), key=lambda classification: k_closest.count(classification))

## To run your classify function on the k-nearest neighbors problem from 2014 Q2
## part B2, uncomment the line below and try different values of k:
# print(knn_classify_point(knn_tree_test_point, knn_tree_data, 1, euclidean_distance))


#### Part 2C: Choosing k #######################################################

def cross_validate(data, k, distance_metric):
    """Given a list of points (the data), an int 0 < k <= len(data), and a
    distance metric (a function), performs leave-one-out cross-validation.
    Return the fraction of points classified correctly, as a float."""
    correct_points = 0
    for point in data:
        training_set = data.copy()
        training_set.remove(point)
        classification = knn_classify_point(point, training_set, k, distance_metric)
        
        if point.classification == classification:
            correct_points += 1

    return correct_points/len(data)
    
def find_best_k_and_metric(data):
    """Given a list of points (the data), uses leave-one-out cross-validation to
    determine the best value of k and distance_metric, choosing from among the
    four distance metrics defined above.  Returns a tuple (k, distance_metric),
    where k is an int and distance_metric is a function."""
    dist_metrics = [euclidean_distance, manhattan_distance, hamming_distance, cosine_distance]
    k = 0
    distance_metric = euclidean_distance
    most_correct = 0

    for metric in dist_metrics:
        for i in range(1, len(data)):
            fraction_correct = cross_validate(data, i+1, metric)
            if fraction_correct > most_correct:
                k = i+1
                distance_metric = metric
                most_correct = fraction_correct
    
    return (k, distance_metric)



## To find the best k and distance metric for 2014 Q2, part B, uncomment:
# print(find_best_k_and_metric(knn_tree_data))


#### Part 2E: More multiple choice #############################################

kNN_ANSWER_1 = "Overfitting"
kNN_ANSWER_2 = "Underfitting"
kNN_ANSWER_3 = 4

kNN_ANSWER_4 = 4
kNN_ANSWER_5 = 1
kNN_ANSWER_6 = 3
kNN_ANSWER_7 = 3


#### SURVEY ####################################################################

NAME = "Peyton Wang"
COLLABORATORS = "Marisa Papagelis"
HOW_MANY_HOURS_THIS_LAB_TOOK = 10
WHAT_I_FOUND_INTERESTING = "Building the tree"
WHAT_I_FOUND_BORING = "Multiple choice"
SUGGESTIONS = "N/A"
