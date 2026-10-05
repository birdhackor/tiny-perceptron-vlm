
Original sklearn-common-pitfalls.rst lines 75-134
.. _data_leakage:

Data leakage
============

Data leakage occurs when information that would not be available at prediction
time is used when building the model. This results in overly optimistic
performance estimates, for example from :ref:`cross-validation
<cross_validation>`, and thus poorer performance when the model is used
on actually novel data, for example during production.

A common cause is not keeping the test and train data subsets separate.
Test data should never be used to make choices about the model.
**The general rule is to never call** `fit` **on the test data**. While this
may sound obvious, this is easy to miss in some cases, for example when
applying certain pre-processing steps.

Although both train and test data subsets should receive the same
preprocessing transformation (as described in the previous section), it is
important that these transformations are only learnt from the training data.
For example, if you have a
normalization step where you divide by the average value, the average should
be the average of the train subset, **not** the average of all the data. If the
test subset is included in the average calculation, information from the test
subset is influencing the model.

How to avoid data leakage
-------------------------

Below are some tips on avoiding data leakage:

* Always split the data into train and test subsets first, particularly
  before any preprocessing steps.
* Never include test data when using the `fit` and `fit_transform`
  methods. Using all the data, e.g., `fit(X)`, can result in overly optimistic
  scores.

  Conversely, the `transform` method should be used on both train and test
  subsets as the same preprocessing should be applied to all the data.
  This can be achieved by using `fit_transform` on the train subset and
  `transform` on the test subset.
* The scikit-learn :ref:`pipeline <pipeline>` is a great way to prevent data
  leakage as it ensures that the appropriate method is performed on the
  correct data subset. The pipeline is ideal for use in cross-validation
  and hyper-parameter tuning functions.

An example of data leakage during preprocessing is detailed below.

Data leakage during pre-processing
----------------------------------

.. note::
    We here choose to illustrate data leakage with a feature selection step.
    This risk of leakage is however relevant with almost all transformations
    in scikit-learn, including (but not limited to)
    :class:`~sklearn.preprocessing.StandardScaler`,
    :class:`~sklearn.impute.SimpleImputer`, and
    :class:`~sklearn.decomposition.PCA`.

A number of :ref:`feature_selection` functions are available in scikit-learn.
