# Project Based Assessment 

The project based assessment library allows the practitioner to estimate difficulty and ability parameters when using data from rubric rows. 

The library contains the following methods:

* getResults
* DisplayResults
* SaveResults

getResults and DisplayResults take the following parameters: 

1. A pandas Dataset containing the columns "k", "student", "rubric", "bound".  The "k" column is the rubric level the given student reached on the given rubric row. The "student" column is a student identifier. The "rubric" column is a rubric row identifier. The "bound" column is maximum "k" value possible on the given rubric row.
2. A float between 0 and 0.5 indicating the portion of the bootstrapped EDF to extract.  For instance, specifying 0.025 would produce the 95% confidence interval. Default is 0.025.
3. A bool flag indicating to treat the rubric rows as blocks instead of the unique students in the bootstrap.  Defaults to False.
4. The number of iterations in the bootstrap.  Defaults to 1000.
5. Uses a simple linear combination of the rubric and student items instead of a sigmoid function when set to true.  Defaults to False.

SaveResults takes the same parameters as getResults and DisplayResults but has the additional parameters of: rubricFile, studentFile, and outputFile (in that order).  These specify the filenames to save the results.  These default to "rubric.csv", "student.csv", and "output.csv".

All methods return the following:

1. Rubric difficulty estimates as a pandas dataframe. Additional interpretation columns are provided in this dataframe that will be described below.
2. Student ability estimates as a pandas dataframe.  Additional interpretation columns are provided in this dataframe that will be described below.
3. Bootstrap confidence intervals and P-Values as a pandas dataframe. P-Values are only provided when estimating the non-linear model as they will always be zero for the linear model (by construction the estimates are constrained between 0 and 1 in the linear model).
4. The number of times the bootstrap routine could not find a solution (if any).
5. Number of observations.
6. Number of parameters.
7. Akaike information criterion
8. Bayesian information criterion
9. McFadden pseudo-R^2
10. Likelihood Ratio test statistic
11. Chi-Squared P-Value of the model (i.e. Wilks' theorem)
12. Log Likelihood value

getResults only return these values as a tuple.  DisplayResults returns the values as a tuple and prints the results to screen.  SaveResults returns the values as a tuple, displays the results and saves the results to CSV files.

The rubric difficulty and student ability pandas dataframes return estimates along with columns used for interpretation.  The following columns are provided: 

* AME k=i: The average marginal effect of k=i.  This is provided for all possible bins (i between 0 and the highest bin).  This procedure calculates the marginal effect for a given estimate conditioned on k=i for all observations impacted by the estimate.  The average is then calculated. These values will sum to zero.
* ACP k=i: While average marginal effect is the standard approach to interpreting MLE results (especially in a logit or probit context), we don't think they are particularly useful in this model.  Therefore, the application also provides columns for the average conditional probability of k=i.  Given the subset of the data used to calculate AME, this is the average probability of k=i given the estimated value.  When the dataset is balanced (all students have a score for all rubric rows), these values will sum to 1.  Note that the top bin is capturing the censoring effect. Therefore, it is common that a substantial probability is estimated for this bin.
* Average Logistic: This estimate is only provided when estimating the non-linear model.  It is the average of the probability function given the estimated value.  It uses the same subset of the data used to calculate AME and ACP above.  In terms of interpretation, it is the average probability of failure to proceed to the next bin.  Therefore, it will equal ACP k=0.
* Average Marginal Logistic: This estimate is only provided when estimating the non-linear model.  It is the average of the marginal probability function given the estimated value.  It uses the same subset of the data used to calculate AME and ACP above.  In terms of interpretation, it is the change in the average probability of failure to proceed to the next bin.

## Background and Use

This package is based on the estimator presented in "[Assessing Proxies of Knowledge and Difficulty with Rubric-Based Instruments](https://dx.doi.org/10.2139/ssrn.4194935)."  There is a [video](https://vimeo.com/735183858) demonstrating using this package in Google Colab and a [video](https://vimeo.com/756447388) explaining the paper.   

## Installation

You can install the package from either PyPI or the Conda repository:

```console
pip install ProjectAssessment
```
or

```console
conda install -c tazzben projectassessment
```
    