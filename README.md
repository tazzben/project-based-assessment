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
4. The number of iterations in the bootstrap.  Defaults to 10000.

SaveResults takes the same parameters as getResults and DisplayResults but has the additional parameters of: rubricFile, studentFile, and outputFile (in that order).  These specify the filenames to save the results.  These default to "rubric.csv", "student.csv", and "output.csv".

All methods return the following:

1. Rubric difficulty estimates as a pandas dataframe.
2. Student ability estimates as a pandas dataframe.
3. Bootstrap confidence intervals and P-Values as a pandas dataframe.
4. The number of times the bootstrap routine could not find a solution (if any).
5. Number of observations.
6. Number of parameters.
7. Akaike information criterion
8. Bayesian information criterion
9. McFadden pseudo-R^2
10. Likelihood Ratio test statistic
11. Chi-Squared P-Value of the model (i.e. Wilks' theorem)

getResults only return these values as a tuple.  DisplayResults returns the values as a tuple and prints the results to screen.  SaveResults returns the values as a tuple, displays the results and saves the results to CSV files.