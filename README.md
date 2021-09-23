# Black Swan

Statistical program for hypothesis testing and data analysis.

Own mini-pandas


formászd át a base permutation class-t, hogy bootstrapp CI-t is adjon a test stat mellé

class parametric:
	...
	pvalue
	CI
	SE

class kappa:
	test_stat


---

Statistical library for hypothesis testing and data analysis.

In progress.


Input:
    .xlsx or .csv file

Returns 
    Word file (or .txt) with results.




I. Descriptive Statistics

- Identify Data types for all columns
- Use pandas, numpy library for Data Analysis
- Store data in a .json file
- Basic metrics (e.g. z-score)

1. Univariate Data

- Compute descriptive statistics 
- Visualize variables
- Comparing CDFs

2. Multivariate Data

- 

3. Visualization

- Use seaborn, matplotlib for automated data viz


II. Inferential Statistics

Check for normality

- Kolmogorov-smirnov
- Shapiro wilk
- Box-plot
- Q-Q plot
- Skewness, Kurtosis
- Histogram


1. Parametric Tests

Uses chi-square, t- and other distribution to test whether the difference between test statistic is how likely to occur by random chance.

- T-test (One-sample, Independent samples)
- Correlation (Pearson, Covariance)
- Chi-square test (Contingency, One way tables)
- Mann Whitney U
- Confidence Interval (for the mean and difference between means, Bootstrap CI)
- Effect size
- Linear Regression

2. Non-parametric Tests

- Hypothesis test (Permutation Test (also known Randomization Test))
- Correlation (Spearman, Correlation that inherits from `PermutationTest` parent class that makes no distributional assumptions)
- Mann-Whitney U (sum ranks, median)
- Kruskal-Wallis (TBD)


When a variable is non-normally distributed (that is, do not resemble a bell-shape), tests assuming normality can have particularly lower power when there are extreme values or outliers. Because non-parametric tests (also known distribution-free tests) do not assume normality, they are less susceptible to non-normality and extreme values. 






# Kaizen

Descriptive Statistics

- Identify Data types for all columns
- Store data in a .json file

Univariate Data

- Compute descriptive statistics based on column information
- Visualize Variables
- 

Multivariate Data

- Groupby columns based on other columns



Inferential Statistics







Visualization
