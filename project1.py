from string import punctuation, digits
import numpy as np
import random
import pandas as pd



#==============================================================================
#===  PART I  =================================================================
#==============================================================================



def get_order(n_samples):
    try:
        with open(str(n_samples) + '.txt') as fp:
            line = fp.readline()
            return list(map(int, line.split(',')))
    except FileNotFoundError:
        random.seed(1)
        indices = list(range(n_samples))
        random.shuffle(indices)
        return indices




def hinge_loss_single(feature_vector, label, theta, theta_0):
    """
    Finds the hinge loss on a single data point given specific classification
    parameters.

    Args:
        `feature_vector` - numpy array describing the given data point.
        `label` - float, the correct classification of the data
            point.
        `theta` - numpy array describing the linear classifier.
        `theta_0` - float representing the offset parameter.
    Returns:
        the hinge loss, as a float, associated with the given data point and
        parameters.
    """


    return np.max([0,1 - label*(np.inner(feature_vector,theta) + theta_0)])






def hinge_loss_full(feature_matrix, labels, theta, theta_0):
    """
    Finds the hinge loss for given classification parameters averaged over a
    given dataset

    Args:
        `feature_matrix` - numpy matrix describing the given data. Each row
            represents a single data point.
        `labels` - numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        `theta` - numpy array describing the linear classifier.
        `theta_0` - real valued number representing the offset parameter.
    Returns:
        the hinge loss, as a float, associated with the given dataset and
        parameters.  This number should be the average hinge loss across all of

    """

    result_hinge_total = 0

    for feacture_vector,label in zip(feature_matrix,labels):
        result_hinge_total += hinge_loss_single(feacture_vector,label,theta,theta_0)

    return result_hinge_total/len(labels)

def perceptron_single_step_update(
        feature_vector,
        label,
        current_theta,
        current_theta_0):
    """
    Updates the classification parameters `theta` and `theta_0` via a single
    step of the perceptron algorithm.  Returns new parameters rather than
    modifying in-place.

    Args:
        feature_vector - A numpy array describing a single data point.
        label - The correct classification of the feature vector.
        current_theta - The current theta being used by the perceptron
            algorithm before this update.
        current_theta_0 - The current theta_0 being used by the perceptron
            algorithm before this update.
    Returns a tuple containing two values:
        the updated feature-coefficient parameter `theta` as a numpy array
        the updated offset parameter `theta_0` as a floating point number
    """
    # Your code here
    condition = label*(np.dot(feature_vector,current_theta) + current_theta_0)
    epsilon = 10**(-15)
    if(condition <= np.abs(epsilon)):
        current_theta = current_theta +  label*feature_vector
        current_theta_0 = current_theta_0 + float(label)
    return (current_theta,current_theta_0)




def perceptron(feature_matrix, labels, T):
    """

    Args:
        `feature_matrix` - numpy matrix describing the given data. Each row
            represents a single data point.
        `labels` - numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        `T` - integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix.

    Returns a tuple containing two values:
        the feature-coefficient parameter `theta` as a numpy array
            (found after T iterations through the feature matrix)
        the offset parameter `theta_0` as a floating point number
            (found also after T iterations through the feature matrix).
    """

    nsamples = feature_matrix.shape[0]
    current_theta_0 = 0.0
    current_theta = np.zeros((feature_matrix.shape[1],), dtype=int)


    for t in range(T):
        for i in get_order(nsamples):
            (current_theta,current_theta_0) = perceptron_single_step_update(feature_matrix[i],labels[i],current_theta,current_theta_0)

    return (current_theta,current_theta_0)




def average_perceptron(feature_matrix, labels, T):
    """
    Runs the average perceptron algorithm on a given dataset.  Runs `T`
    iterations through the dataset (we do not stop early) and therefore
    averages over `T` many parameter values.



    Args:
        `feature_matrix` -  A numpy matrix describing the given data. Each row
            represents a single data point.
        `labels` - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        `T` - An integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix.

    Returns a tuple containing two values:
        the average feature-coefficient parameter `theta` as a numpy array
            (averaged over T iterations through the feature matrix)
        the average offset parameter `theta_0` as a floating point number
            (averaged also over T iterations through the feature matrix).
    """

    nsamples = feature_matrix.shape[0]
    current_theta_0 = 0.0
    current_theta = np.zeros((feature_matrix.shape[1],), dtype=int)
    sum_thetas = 0
    sum_thetas_zeros = 0


    for t in range(T):
        for i in get_order(nsamples):
            (current_theta,current_theta_0) = perceptron_single_step_update(feature_matrix[i], labels[i], current_theta, current_theta_0)
            sum_thetas += current_theta
            sum_thetas_zeros += current_theta_0

    return (sum_thetas/(nsamples*T),sum_thetas_zeros/(nsamples*T))


def pegasos_single_step_update(
        feature_vector,
        label,
        L,
        eta,
        theta,
        theta_0):

    if (label*(np.inner(theta,feature_vector) + theta_0)<=1):
        theta = (1 - eta*L)*theta + eta*label*feature_vector
        theta_0 += eta*label
        return(theta,theta_0)

    elif(label*(np.inner(theta,feature_vector) + theta_0)>1):
        theta = (1 - eta*L)*theta
        return(theta,theta_0)









def pegasos(feature_matrix, labels, T, L):
    """

    Args:
        `feature_matrix` - A numpy matrix describing the given data. Each row
            represents a single data point.
        `labels` - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        `T` - An integer indicating how many times the algorithm
            should iterate through the feature matrix.
        `L` - The lamba value being used to update the Pegasos
            algorithm parameters.

    Returns:
        a tuple where the first element is a numpy array with the value of the
        theta, the linear classification parameter, found after T iterations
        through the feature matrix and the second element is a real number with
        the value of the theta_0, the offset classification parameter, found
        after T iterations through the feature matrix.
    """

    theta = np.zeros((feature_matrix.shape[1],))
    theta_0 = 0
    up = 0
    for t in range(T):
        for i in get_order(feature_matrix.shape[0]):
            up +=1
            (theta,theta_0) = pegasos_single_step_update(feature_matrix[i],labels[i],L,1/np.sqrt(up),theta,theta_0)

    return (theta,theta_0)




#==============================================================================
#===  PART II  ================================================================
#==============================================================================



##  #pragma: coderesponse answer
##  def decision_function(feature_vector, theta, theta_0):
##      return np.dot(theta, feature_vector) + theta_0
##  def classify_vector(feature_vector, theta, theta_0):
##      return 2*np.heaviside(decision_function(feature_vector, theta, theta_0), 0)-1
##  #pragma: coderesponse end



def classify(feature_matrix, theta, theta_0):
    """
    A classification function that uses given parameters to classify a set of
    data points.

    Args:
        `feature_matrix` - numpy matrix describing the given data. Each row
            represents a single data point.
        `theta` - numpy array describing the linear classifier.
        `theta_0` - real valued number representing the offset parameter.

    Returns:
        a numpy array of 1s and -1s where the kth element of the array is the
        predicted classification of the kth row of the feature matrix using the
        given theta and theta_0. If a prediction is GREATER THAN zero, it
        should be considered a positive classification.
    """

    h_vector = np.array([])
    epsilon = 10**(-16)
    for feature_vector in feature_matrix:
        if(np.inner(theta,feature_vector)+theta_0 >= epsilon):
            h_vector = np.append(h_vector,1)
        elif(np.inner(theta,feature_vector)+theta_0 < epsilon):
            h_vector = np.append(h_vector,-1)
    return(h_vector)




def classifier_accuracy(
        classifier,
        train_feature_matrix,
        val_feature_matrix,
        train_labels,
        val_labels,
        **kwargs):
    """
    Trains a linear classifier and computes accuracy.  The classifier is
    trained on the train data.  The classifier's accuracy on the train and
    validation data is then returned.

    Args:
        `classifier` - A learning function that takes arguments
            (feature matrix, labels, **kwargs) and returns (theta, theta_0)
        `train_feature_matrix` - A numpy matrix describing the training
            data. Each row represents a single data point.
        `val_feature_matrix` - A numpy matrix describing the validation
            data. Each row represents a single data point.
        `train_labels` - A numpy array where the kth element of the array
            is the correct classification of the kth row of the training
            feature matrix.
        `val_labels` - A numpy array where the kth element of the array
            is the correct classification of the kth row of the validation
            feature matrix.
        `kwargs` - Additional named arguments to pass to the classifier
            (e.g. T or L)

    Returns:
        a tuple in which the first element is the (scalar) accuracy of the
        trained classifier on the training data and the second element is the
        accuracy of the trained classifier on the validation data.
    """

    #First step : Training

    (theta,theta_0) = classifier(train_feature_matrix,train_labels, **kwargs)

    #Second step : Classification
    train_preds = classify(train_feature_matrix,theta,theta_0)
    val_preds = classify(val_feature_matrix,theta,theta_0)

    #Third step : Accuracy
    train_accuracy = accuracy(train_preds,train_labels)
    val_accuracy = accuracy(val_preds,val_labels)

    return(train_accuracy,val_accuracy)





def extract_words(text):

    for c in punctuation + digits:
        text = text.replace(c, ' ' + c + ' ')
    return text.lower().split()



def bag_of_words(texts, remove_stopword=False):


    stopword = pd.read_table('stopwords.txt',header = None)
    stopword = stopword.values.reshape([127])


    indices_by_word = {}  # maps word to unique index

    for text in texts:
        word_list = extract_words(text)
        for word in word_list:
            if word in indices_by_word: continue
            if word in stopword: continue
            indices_by_word[word] = len(indices_by_word)

    return indices_by_word



def extract_bow_feature_vectors(reviews, indices_by_word, binarize=False):
    """
    Args:
        `reviews` - a list of natural language strings
        `indices_by_word` - a dictionary of uniquely-indexed words.
    Returns:
        a matrix representing each review via bag-of-words features.  This
        matrix thus has shape (n, m), where n counts reviews and m counts words
        in the dictionary.
    """


    feature_matrix = np.zeros([len(reviews), len(indices_by_word)], dtype=np.float64)
    for i, text in enumerate(reviews):
        word_list = extract_words(text)
        for word in word_list:
            if word not in indices_by_word: continue
            feature_matrix[i, indices_by_word[word]] += 1

    return feature_matrix



def accuracy(preds, targets):
    """
    Given length-N vectors containing predicted and target labels,
    returns the fraction of predictions that are correct.
    """
    return (preds == targets).mean()
