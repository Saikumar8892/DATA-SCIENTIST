import pandas as pd  # Importing pandas library for data manipulation
import numpy as np  # Importing numpy library for numerical operations
import matplotlib.pyplot as plt  # Importing matplotlib for data visualization
import scipy  # Importing scipy for scientific computing
from scipy import stats  # Importing stats module from scipy for statistical functions
import statsmodels.stats.descriptivestats as sd  # Importing stats module from statsmodels for descriptive statistics
from statsmodels.stats import weightstats as stests  # Importing weightstats module from statsmodels for statistical tests

############# 1-Sample Z-Test #############
# Business Problem: Verify if the length of the fabric is being cut at appropriate sizes (lengths)

# importing the data
fabric = pd.read_csv(r"C:\Users\Bharani Kumar\Desktop\Data Science using Python & R\Version 2 slides\Datasets\Hypothesis_datasets\Fabric_data.csv")

# Calculating the normality test
# Hypothesis: Ho = Data are Normal, Ha = Data are not Normal
print(stats.shapiro(fabric.Fabric_length))  # Shapiro-Wilk test for normality
# p-value = 0.146 > 0.05 so p high null fly => Data are Normal

# Calculate the mean of the Fabric_length column in the fabric DataFrame
np.mean(fabric.Fabric_length)

# Population standard deviation (Sigma) is known
# z-test
# Ho: Current Mean is Equal to Standard Mean (150) => No action
# Ha: Current Mean is Not Equal to Standard Mean (150) => Take action


# Perform a z-test to compare the mean of Fabric_length against a specified value (150)
# stests.ztest is assumed to be a function from the statsmodels.stats library for conducting z-tests
# The function takes parameters: 
#   - fabric.Fabric_length: the data to be tested
#   - x2=None: there is no second sample to compare against
#   - value=150: the null hypothesis value against which the mean is tested
ztest, pval = stests.ztest(fabric.Fabric_length, x2=None, value=150)

# Print the p-value obtained from the z-test
print(float(pval))

# p-value = 7.156e-06 < 0.05 so p low null go

# z-test
# parameters in z-test, value is mean of data
# z-test
# Ho: Current Mean <= Standard Mean (150) => No action
# Ha: Current Mean > Standard Mean (150) => Take action

# Perform a one-sample z-test to compare the mean of the Fabric_length column in the fabric DataFrame against a specified value (150)
# The alternative hypothesis is that the mean is greater than the specified value (150)
# stests.ztest is assumed to be a function from the statsmodels.stats library for conducting z-tests
# The function takes parameters: 
#   - fabric.Fabric_length: the data to be tested
#   - x2=None: there is no second sample to compare against
#   - value=150: the null hypothesis value against which the mean is tested
#   - alternative='larger': specifying the alternative hypothesis as 'larger' means testing if the mean is greater than the specified value
ztest, pval = stests.ztest(fabric.Fabric_length, x2=None, value=150, alternative='larger')

# Convert the p-value obtained from the z-test to a float and print it
print(float(pval))


# p-value = 3.578-06 < 0.05 => p low null go

# Conclusion: Stop the production and verify the problem with the machine

############# 1-Sample t-Test #############
# Business Problem: Verify if the monthly energy cost differs from $200.

# loading the csv file
data = pd.read_csv(r"C:\Users\Bharani Kumar\Desktop\Data Science using Python & R\Version 2 slides\Datasets\Hypothesis_datasets\FamilyEnergyCost.csv")

# Display descriptive statistics of the 'data' DataFrame using the describe() method
data.describe()

# Normality test
# Hypothesis: Ho = Data are Normal, Ha = Data are not Normal
print(stats.shapiro(data['Energy Cost']))  # Shapiro-Wilk test for normality
# p-value = 0.764 > 0.05 so p high null fly => Data are Normal

# Population standard deviation (Sigma) is not known

# Perform a one-sample t-test to compare the mean of the 'Energy Cost' column in the 'data' DataFrame against a specified population mean of 200
# The default alternative hypothesis is two-sided (mean != popmean)
# stats.ttest_1samp is assumed to be a function from the scipy.stats library for conducting t-tests
# The function takes parameters:
#   - a=data['Energy Cost']: the data to be tested
#   - popmean=200: the population mean against which the mean is tested
# This line calculates the p-value of the two-tailed test and assigns it to the variable 'p_value'
t_statistic, p_value = stats.ttest_1samp(a=data['Energy Cost'], popmean=200)
# Print the p-value rounded to 2 decimal places
print("%.2f" % p_value)

# Perform a one-sample t-test with the alternative hypothesis that the mean of the 'Energy Cost' column is greater than the specified population mean of 200
# The alternative hypothesis is one-sided (mean > popmean)
# This line calculates the p-value of the right-tailed test and assigns it to the variable 'p_value'
t_statistic, p_value = stats.ttest_1samp(a=data['Energy Cost'], popmean=200, alternative='greater')
# Print the p-value rounded to 2 decimal places
print("%.2f" % p_value)

# p-value = 0.00 < 0.05 => p low null go

# Conclusion: The average monthly energy cost for families is greater than $200. So reduce electricity consumption

############ Non-Parameteric Test ############

############ 1 Sample Sign Test ################
# Stainless-Steel Chromium content
steel = pd.read_excel(r"C:\Users\Bharani Kumar\Desktop\Data Science using Python & R\Version 2 slides\Datasets\Hypothesis_datasets\StainlessSteelComposition.xlsx")
steel

# Note: Most stainless steel contains about 18 percent chromium;
# it is what hardens and toughens steel and increases its resistance
# to corrosion, especially at high temperatures.

# Business Problem: Determine whether the median chromium content differs from 18%.

# Normality Test
# Ho = Data are Normal
# Ha = Data are not Normal
stats.shapiro(steel.Chromium)  # Shapiro Test

# p-value = 0.00016 < 0.05 so p low null go => Data are not Normal

### 1 Sample Sign Test ###

# Ho: The median chromium content is equal to 18%.
# Ha: The median chromium content is not equal to 18%.

sd.sign_test(steel.Chromium, mu0=18)
# sd.sign_test(marks.Scores, mu0 = marks.Scores.median())

# Conclusion: Enough evidence to conclude that the median chromium content is equal to 18%.

######### Mann-Whitney Test ############
# Vehicles with and without additive

# Business Problem: Fuel additive is enhancing the performance (mileage) of a vehicle

# Read the CSV file 'mann_whitney_additive.csv' located at the specified file path into a DataFrame and assign it to the variable 'fuel'
# The 'r' prefix before the file path signifies a raw string, which prevents Python's escape sequences from being interpreted
fuel = pd.read_csv(r"C:\Users\Bharani Kumar\Desktop\Data Science using Python & R\Version 2 slides\Datasets\Hypothesis_datasets\mann_whitney_additive.csv")

# Display the contents of the 'fuel' DataFrame to view the data
fuel

# Rename the columns of the 'fuel' DataFrame to "Without_additive" and "With_additive"
# This renames the column labels to provide clear and descriptive names for the data
fuel.columns = "Without_additive", "With_additive"


# Normality test
# Ho = Data are Normal
# Ha = Data are not Normal

# Perform the Shapiro-Wilk test for normality on the 'Without_additive' column of the 'fuel' DataFrame
# Print the result, including the test statistic and p-value
# Comment: This test checks the null hypothesis that the data in the 'Without_additive' column is normally distributed.
# If the p-value is high (greater than the chosen significance level), we fail to reject the null hypothesis, suggesting normality.
print(stats.shapiro(fuel.Without_additive))


# Perform the Shapiro-Wilk test for normality on the 'With_additive' column of the 'fuel' DataFrame
# Print the result, including the test statistic and p-value
# Comment: This test checks the null hypothesis that the data in the 'With_additive' column is normally distributed.
# If the p-value is low (less than the chosen significance level), we reject the null hypothesis, suggesting non-normality.
print(stats.shapiro(fuel.With_additive))  # p low null go

# Data are not normal

# Non-Parameteric Test case
# Mann-Whitney test

# Ho: Mileage with and without Fuel additive is the same
# Ha: Mileage with and without Fuel additive are different

# Perform the Mann-Whitney U test for comparing two independent samples from non-normally distributed populations
# The test compares the medians of the two samples and is used when assumptions of normality and equal variance are not met
# Comment: This test evaluates whether there is a statistically significant difference between the distributions of 'Without_additive' and 'With_additive' samples.
# The result includes the test statistic and p-value, which indicate the strength of evidence against the null hypothesis of no difference.
scipy.stats.mannwhitneyu(fuel.Without_additive, fuel.With_additive)

# p-value = 0.44 > 0.05 so p high null fly
# Ho: fuel additive does not impact the performance

############### Paired T-Test ##############
# A test to determine whether there is a significant difference between 2 variables.

# Data:
#  Data shows the effect of two soporific drugs
# (increase in hours of sleep compared to control) on 10 patients.
# External Conditions are conducted in a controlled environment to ensure external conditions are same

# Business Problem: Determine which of the two soporific drugs increases the sleep duration

sleep = pd.read_csv(r"C:\Users\Bharani Kumar\Desktop\Data Science using Python & R\Version 2 slides\Datasets\Hypothesis_datasets\sleep.csv")
sleep.describe()

# Perform the Shapiro-Wilk test for normality on the first 10 observations of the 'extra' column in the 'sleep' DataFrame
# Print the result, including the test statistic and p-value
# Comment: This test checks the null hypothesis that the first 10 observations of the 'extra' column are normally distributed.
stats.shapiro(sleep.extra[0:10])

# Perform the Shapiro-Wilk test for normality on the next 10 observations of the 'extra' column in the 'sleep' DataFrame
# Print the result, including the test statistic and p-value
# Comment: This test checks the null hypothesis that the next 10 observations of the 'extra' column are normally distributed.
stats.shapiro(sleep.extra[10:20])

# Perform a paired t-test (related samples t-test) on the first 10 observations and the next 10 observations of the 'extra' column in the 'sleep' DataFrame
# Print the p-value obtained from the test
# Comment: This test evaluates whether there is a statistically significant difference between the means of paired observations from two related groups.
# The p-value indicates the probability of observing the obtained results if the null hypothesis of equal means were true.
ttest, pval = stats.ttest_rel(sleep.extra[0:10], sleep.extra[10:20])
print(pval)
# p-value = 0.002 < 0.05 => p low null go
# Ha: Increase in the sleep with Drug 1 != Increase in the sleep with Drug 2

# Ho: Increase in the sleep with Drug 1 <= Increase in the sleep with Drug 2
# Ha: Increase in the sleep with Drug 1 > Increase in the sleep with Drug 2
ttest, pval = stats.ttest_rel(sleep.extra[0:10], sleep.extra[10:20], alternative='greater')
print(pval)

# p-value = 0.99 > 0.05 => p high null fly
# Ho: Increase in the sleep with Drug 1 <= Increase in the sleep with Drug 2

############ 2-sample t-Test (Marketing Strategy) ##################
# Business Problem: Determine whether the Full Interest Rate Waiver is better than Standard Promotion

# Load the data from the Excel file into a DataFrame named 'prom'
prom = pd.read_excel(r"C:\Users\Bharani Kumar\Desktop\Data Science using Python & R\Version 2 slides\Datasets\Hypothesis_datasets\Promotion.xlsx")

# Rename the columns of the DataFrame to 'InterestRateWaiver' and 'StandardPromotion'
prom.columns = "InterestRateWaiver", "StandardPromotion"

# Perform Shapiro-Wilk tests for normality on the 'InterestRateWaiver' and 'StandardPromotion' columns
# H0 = Data are normally distributed
# Ha = Data are not normally distributed
shapiro_irw = stats.shapiro(prom.InterestRateWaiver)  # Shapiro-Wilk test for 'InterestRateWaiver'
shapiro_sp = stats.shapiro(prom.StandardPromotion)    # Shapiro-Wilk test for 'StandardPromotion'
print(shapiro_irw)
print(shapiro_sp)
# Comments: The p-values obtained from the Shapiro-Wilk tests indicate that both datasets are normally distributed.

# Perform Levene's test for equality of variances
# H0 = Variances are equal
# Ha = Variances are not equal
levene_test = scipy.stats.levene(prom.InterestRateWaiver, prom.StandardPromotion)
# p-value = 0.287 > 0.05, so we fail to reject the null hypothesis, indicating equal variances.
print(levene_test)

# Perform two-sample independent t-tests to compare the mean purchases between the two promotion methods
# H0: Average purchases due to both promotions are equal
# Ha: Average purchases due to both promotions are unequal
t_test = scipy.stats.ttest_ind(prom.InterestRateWaiver, prom.StandardPromotion)
# p-value = 0.024 < 0.05, rejecting the null hypothesis, suggesting unequal average purchases.
print(t_test)

# Perform a one-sided two-sample independent t-test with alternative='greater'
# H0: Average purchases due to InterestRateWaiver <= Average purchases due to StandardPromotion
# Ha: Average purchases due to InterestRateWaiver > Average purchases due to StandardPromotion
t_test_greater = scipy.stats.ttest_ind(prom.InterestRateWaiver, prom.StandardPromotion, alternative='greater')
# p-value = 0.012 < 0.05, indicating that the average purchases due to InterestRateWaiver are significantly greater.
print(t_test_greater)

# Conclusion: Interest Rate Waiver is a more effective promotion strategy compared to Standard Promotion.

###### Moods-Median Test ######
# Business Problem: Determine if the weight of the fish differs with the change in the temperatures.

# Import the dataset 'Fishweights.csv' into a DataFrame named 'fish'
fish = pd.read_csv(r"C:\Users\Bharani Kumar\Desktop\Data Science using Python & R\Version 2 slides\Datasets\Hypothesis_datasets\Fishweights.csv")

# Display the summary statistics of the dataset
fish.describe()

# Split the dataset into four groups based on the 'group' column
g1 = fish[fish.group == 1]  # Group 1
g2 = fish[fish.group == 2]  # Group 2
g3 = fish[fish.group == 3]  # Group 3
g4 = fish[fish.group == 4]  # Group 4

# Perform Shapiro-Wilk tests for normality on each group
# H0 = Data are normally distributed
# Ha = Data are not normally distributed
shapiro_g1 = stats.shapiro(g1.Weight)
shapiro_g2 = stats.shapiro(g2.Weight)
shapiro_g3 = stats.shapiro(g3.Weight)
shapiro_g4 = stats.shapiro(g4.Weight)

# Perform Mood's median test to compare medians among the four groups
# Ho: The population medians are equal across fish groups.
# Ha: The population medians are not equal across fish groups.
from scipy.stats import median_test
stat, p_value, med, tbl = median_test(g1.Weight, g2.Weight, g3.Weight, g4.Weight)

# Print the p-value obtained from the median test
p_value

# 0.696 > 0.05 => P High Null Fly

# Fail to reject the null hypothesis
# The differences between the median weights are not statistically significant.
# Further tests must be done to determine which fish weight is more than the rest, which is out of scope for our discussion.

############# One-Way ANOVA #############
# Business Problem: CMO to determine the renewal of contracts of the suppliers based on their performances

# Import the dataset 'ContractRenewal_Data(unstacked).xlsx' into a DataFrame named 'con_renewal'
con_renewal = pd.read_excel(
    r"C:\Users\Bharani Kumar\Desktop\Data Science using Python & R\Version 2 slides\Datasets\Hypothesis_datasets\ContractRenewal_Data(unstacked).xlsx")

# Display the dataset
con_renewal

# Rename the columns for clarity
con_renewal.columns = "SupplierA", "SupplierB", "SupplierC"

# Perform Shapiro-Wilk tests for normality on each supplier's transaction time data
# H0 = Data are normally distributed
# Ha = Data are not normally distributed
shapiro_supplierA = stats.shapiro(con_renewal.SupplierA)
shapiro_supplierB = stats.shapiro(con_renewal.SupplierB)
shapiro_supplierC = stats.shapiro(con_renewal.SupplierC)

# Perform Levene's test for homogeneity of variances
# Ho: All the 3 suppliers have equal variance in transaction time
# Ha: All the 3 suppliers have unequal variance in transaction time
levene_test = scipy.stats.levene(con_renewal.SupplierA, con_renewal.SupplierB, con_renewal.SupplierC)
# Since p-value > 0.05, variances are statistically equal

# Perform one-way ANOVA to compare the mean transaction time among the three suppliers
# Ho: All the 3 suppliers have equal mean transaction time
# Ha: All the 3 suppliers have unequal mean transaction time
F_statistic, p_value = stats.f_oneway(con_renewal.SupplierA, con_renewal.SupplierB, con_renewal.SupplierC)

# Print the p-value obtained from the ANOVA test
# Since p-value > 0.05, fail to reject the null hypothesis, indicating that all suppliers have equal mean transaction time
p_value


######### 1-Proportion Test #########
# Business Problem: The proportion of smokers is varying with the historical figure of 25 percent of students who smoke regularly.

# Import the dataset 'Smokers.csv' into a DataFrame named 'smokers'
smokers = pd.read_csv(
    r"C:\Users\Bharani Kumar\Desktop\Data Science using Python & R\Version 2 slides\Datasets\Hypothesis_datasets\Smokers.csv")

# Display the first few rows of the dataset
smokers.head()

# Count the number of smokers and non-smokers
observed_counts = smokers['Smokes'].value_counts()
observed_counts

# Extract the number of smokers
x = observed_counts[1]

# Get the total number of observations
n = len(smokers)

# Perform a two-sided z-test for proportions to test the hypothesis
# Ho: Proportion of students who smoke regularly is less than or equal to the historical figure 25%
# Ha: Proportion of students who smoke regularly is greater than the historical figure 25%
from statsmodels.stats.proportion import proportions_ztest
z_statistic, p_value = proportions_ztest(count=x, nobs=n, value=0.25, alternative='two-sided')

# Print the p-value obtained from the z-test
# If p-value > 0.05, fail to reject the null hypothesis (Ho)
print("%.2f" % p_value)
# Since p-value (0.33) > 0.05, fail to reject Ho, indicating no significant increase in the proportion of smokers

######### 2-Proportion Test #########

# Business Problem: Sales manager has to determine if the sales incentive program should be launched or not

import numpy as np

# Load the dataset 'JohnyTalkers.xlsx' into a DataFrame
two_prop_test = pd.read_excel(
    r"C:\Users\Bharani Kumar\Desktop\Data Science using Python & R\Version 2 slides\Datasets\Hypothesis_datasets\JohnyTalkers.xlsx")

# Count the occurrences of each category in the 'Person' column
person_counts = two_prop_test.Person.value_counts()
print(person_counts)

# Count the occurrences of each category in the 'Drinks' column
drinks_counts = two_prop_test.Drinks.value_counts()
print(drinks_counts)

# Create a cross-tabulation table of 'Person' versus 'Drinks'
pd.crosstab(two_prop_test.Person, two_prop_test.Drinks)

# Define the count and number of observations arrays for the z-test
count = np.array([58, 152])  # Counts of 'Adults' and 'Children' who drink
nobs = np.array([480, 740])   # Total observations for 'Adults' and 'Children'

# Case 1: Two-sided test
# Ho: Proportions of Adults = Proportions of Children
# Ha: Proportions of Adults != Proportions of Children
stats, p_value = proportions_ztest(count, nobs, alternative='two-sided')
print("%.2f" % p_value)
# Since p-value (0.000) < 0.05, reject Ho, indicating proportions of Adults and Children are different

# Case 2: One-sided (Greater) test
# Ho: Proportions of Adults <= Proportions of Children
# Ha: Proportions of Adults > Proportions of Children
stats, p_value = proportions_ztest(count, nobs, alternative='larger')
print("%.2f" % p_value)
# Since p-value (1.0) > 0.05, fail to reject Ho, indicating proportions of Adults are not greater than Children

# Conclusion: Do not launch the incentive program as there's no evidence that Adults drink more than Children


############### Chi-Square Test ################
# Business Problem: Check whether the questionnaire responses input entry defectives % varies by region.

# This code analyzes a dataset to see if there's a relationship between defective products and the country they come from

# Read the data from an Excel file
Bahaman = pd.read_excel(
    r"C:\Users\Bharani Kumar\Desktop\Data Science using Python & R\Version 2 slides\Datasets\Hypothesis_datasets\Bahaman.xlsx"
)

# Explore the data (optional)
# Bahaman  # This line can be used to view the data

# Create a contingency table to see how many defective and non-defective products come from each country
count = pd.crosstab(Bahaman["Defective"], Bahaman["Country"])

# Set up the null and alternative hypothesis for the chi-square test
#  - Null hypothesis (H₀): All countries have the same proportion of defectives.
#  - Alternative hypothesis (Ha): Not all countries have the same proportion of defectives.
print("Null hypothesis (H₀): All countries have the same proportion of defectives.")
print("Alternative hypothesis (Ha): Not all countries have the same proportion of defectives.")

# Perform the chi-square test to see if there's a statistically significant difference
Chisquares_results = scipy.stats.chi2_contingency(count)

# Print the test statistic and p-value from the results
print("Chi-Square Test Results:")
print("Test Statistic:", Chisquares_results[0])
print("p-value:", Chisquares_results[1])

# Interpret the p-value (typically a significance level of 0.05 is used)
# - If the p-value is greater than 0.05, we fail to reject the null hypothesis (H₀).
# - If the p-value is less than or equal to 0.05, we reject the null hypothesis (H₀) in favor of the alternative hypothesis (Ha).
print("Based on the p-value, we will decide if we can reject the null hypothesis.")



### The End

