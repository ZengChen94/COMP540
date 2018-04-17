import numpy as np

def estimate_gaussian(X):
    
    """
    Estimate the mean and standard deviation of a numpy matrix X on a column by column basis
    """
    mu = np.zeros((X.shape[1],))
    var = np.zeros((X.shape[1],))
    ####################################################################
    #               YOUR CODE HERE                                     #
    ####################################################################
    mu = np.mean(X, axis=0)
    var = np.var(X, axis=0)
    ####################################################################
    #               END YOUR CODE                                      #
    ####################################################################
    return mu, var


def select_threshold(yval,pval):
    """
    select_threshold(yval, pval) finds the best
    threshold to use for selecting outliers based on the results from a
    validation set (pval) and the ground truth (yval).
    """

    best_epsilon = 0
    bestF1 = 0
    stepsize = (max(pval)-min(pval))/1000
    for epsilon in np.arange(min(pval)+stepsize, max(pval), stepsize):
        
        ####################################################################
        #                 YOUR CODE HERE                                   #
        ####################################################################
#         bestF1 = 0
        pred = pval < epsilon
        tp = np.sum(pred[np.nonzero(yval.ravel() == True)])
        fp = np.sum(pred[np.nonzero(yval.ravel() == False)])
        fn = np.sum(yval[np.nonzero(pred.ravel() == False)] == True)
        prec = 1.0*tp/(tp + fp)
        rec = 1.0*tp/(tp + fn)
        F1 = 2.0*prec*rec/(prec + rec)
        if F1 > bestF1:
            best_epsilon = epsilon
            bestF1 = F1
            

        ####################################################################
        #                 END YOUR CODE                                    #
        ####################################################################
    return best_epsilon, bestF1
