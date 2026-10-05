
.. _cross_validation:

===================================================
Cross-validation: evaluating estimator performance
===================================================

.. currentmodule:: sklearn.model_selection

Learning the parameters of a prediction function and testing it on the
same data is a methodological mistake: a model that would just repeat
the labels of the samples that it has just seen would have a perfect
score but would fail to predict anything useful on yet-unseen data.
This situation is called **overfitting**.
To avoid it, it is common practice when performing
a (supervised) machine learning experiment
to hold out part of the available data as a **test set** ``X_test, y_test``.
Note that the word "experiment" is not intended
to denote academic use only,
because even in commercial settings
machine learning usually starts out experimentally.
Here is a flowchart of typical cross validation workflow in model training.
The best parameters can be determined by
:ref:`grid search <grid_search>` techniques.

.. image:: ../images/grid_search_workflow.png
   :width: 400px
   :height: 240px
   :alt: Grid Search Workflow
   :align: center

In scikit-learn a random split into training and test sets
can be quickly computed with the :func:`train_test_split` helper function.
Let's load the iris data set to fit a linear support vector machine on it::

  >>> import numpy as np
  >>> from sklearn.model_selection import train_test_split
  >>> from sklearn import datasets
  >>> from sklearn import svm

  >>> X, y = datasets.load_iris(return_X_y=True)
  >>> X.shape, y.shape
  ((150, 4), (150,))

We can now quickly sample a training set while holding out 40% of the
data for testing (evaluating) our classifier::

  >>> X_train, X_test, y_train, y_test = train_test_split(
  ...     X, y, test_size=0.4, random_state=0)

  >>> X_train.shape, y_train.shape
  ((90, 4), (90,))
  >>> X_test.shape, y_test.shape
  ((60, 4), (60,))

  >>> clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
  >>> clf.score(X_test, y_test)
  0.96

When evaluating different settings ("hyperparameters") for estimators,
such as the ``C`` setting that must be manually set for an SVM,
there is still a risk of overfitting *on the test set*
because the parameters can be tweaked until the estimator performs optimally.
This way, knowledge about the test set can "leak" into the model
and evaluation metrics no longer report on generalization performance.
To solve this problem, yet another part of the dataset can be held out
as a so-called "validation set": training proceeds on the training set,
after which evaluation is done on the validation set,
and when the experiment seems to be successful,
final evaluation can be done on the test set.

However, by partitioning the available data into three sets,
we drastically reduce the number of samples
which can be used for learning the model,
and the results can depend on a particular random choice for the pair of
(train, validation) sets.

A solution to this problem is a procedure called
`cross-validation <https://en.wikipedia.org/wiki/Cross-validation_(statistics)>`_
(CV for short).
A test set should still be held out for final evaluation,
but the validation set is no longer needed when doing CV.
In the basic approach, called *k*-fold CV,
the training set is split into *k* smaller sets
(other approaches are described below,
but generally follow the same principles).
The following procedure is followed for each of the *k* "folds":

* A model is trained using :math:`k-1` of the folds as training data;
* the resulting model is validated on the remaining part of the data
  (i.e., it is used as a test set to compute a performance measure
  such as accuracy).

The performance measure reported by *k*-fold cross-validation
is then the average of the values computed in the loop.
This approach can be computationally expensive,
but does not waste too much data
(as is the case when fixing an arbitrary validation set),
which is a major advantage in problems such as inverse inference
where the number of samples is very small.

.. image:: ../images/grid_search_cross_validation.png
   :width: 500px
   :height: 300px
   :alt: A depiction of a 5 fold cross validation on a training set, while holding out a test set.
