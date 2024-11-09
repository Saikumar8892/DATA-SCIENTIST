#Hypothesis testing is a statistical method used to make inferences or draw conclusions about a population based on sample data.
#It helps to decide if there is enough evidence to support a specific claim or hypothesis about a population parameter.
#Problem statement:-Difference in Diameter of Cutlets Between Two Units
#Objective: To determine if there is a significant difference in the diameter of cutlets between Unit A and Unit B.
#Importing Libraries
import pandas as pd#Used for data handling and manipulation in DataFrames
from scipy.stats import chi2_contingency#Performs a chi-square test for independence
from scipy.stats import shapiro, levene, ttest_ind
#shapiro: Shapiro-Wilk test for normality.
#levene: Levene’s test for equality of variances.
#ttest_ind: Two-sample t-test for the means of two independent samples
from scipy.stats import f_oneway#One-way ANOVA test, useful for comparing means across multiple groups.
# Load the dataset
cutlets_df = pd.read_csv('Cutlets.csv')#reads the CSV file and stores it in cutlets_df.
# Display the first few rows and summary information about the dataset
cutlets_df.head()# shows the first few rows
cutlets_df.describe()#provides summary statistics (like mean, std, min, max) for numerical columns.
cutlets_df.info()#prints info about columns, including data types and missing values.
# Extract data for each unit
#Extracts the columns Unit A and Unit B, removing any missing values (NaN values).
unit_a = cutlets_df['Unit A'].dropna()#unit_a are separate datasets for analysis
unit_b = cutlets_df['Unit B'].dropna()#unit_b are separate datasets for analysis
# Normality Test (Shapiro-Wilk) for each unit
shapiro_a = shapiro(unit_a)
shapiro_b = shapiro(unit_b)
#Shapiro-Wilk Test: The Shapiro-Wilk test checks for normality. shapiro_a and shapiro_b store the test statistic and p-value for unit_a and unit_b.
#If p > 0.05, data is likely normally distributed.
# Variance Test (Levene's test)
levene_test = levene(unit_a, unit_b)
#Levene's Test: This test checks if unit_a and unit_b have equal variances, an assumption for the two-sample t-test.
#If p > 0.05, the variances are likely equal.
# Independent Two-Sample T-Test
t_test = ttest_ind(unit_a, unit_b, equal_var=True)
#Two-Sample T-Test: Tests if the means of unit_a and unit_b are significantly different, assuming equal variances (equal_var=True).
#If p > 0.05, no significant difference in means is inferred.
shapiro_a, shapiro_b, levene_test, t_test
#Returns the results of Shapiro-Wilk, Levene’s, and t-test for review.
#Problem 3: Buyer Ratio Across Regions
#Objective: To determine if the male-to-female buyer ratios are similar across the regions (East, West, North, South).
data = pd.read_csv('BuyerRatio.csv')
# Display the data to understand its structure
data#Prints the dataset to understand its structure and contents.
# Prepare data for chi-square test
# Extract only the numeric part of the table
observed_values = data.iloc[:, 1:].values  # exclude the first column which has row labels
#Extract Observed Values: Selects only the numeric part of the table (all columns except the first one), storing these as observed_values for chi-square testing.
# Perform the Chi-Square test
chi2_stat, p_value, dof, expected_values = chi2_contingency(observed_values)
#Chi-Square Test: Tests for independence between categorical variables.
#chi2_stat: Chi-square statistic.
#p_value: Determines if there’s significant association.
#dof: Degrees of freedom.
#expected_values: Expected frequency table if there were no association.
# Output the chi-square test statistic, p-value, degrees of freedom, and expected values
chi2_stat, p_value, dof, expected_values
#Problem3: Average Turn Around Time (TAT) Among Four Laboratories
#Objective: To determine if there is a difference in the average TAT among the four laboratories.
# Load the data from the CSV file
data = pd.read_csv('lab_tat.csv')
# Display the first few rows of the dataset to understand its structure
data.head()
# Perform ANOVA test across the four laboratories
anova_result = f_oneway(data['Laboratory_1'], 
                        data['Laboratory_2'], 
                        data['Laboratory_3'], 
                        data['Laboratory_4'])
#ANOVA Test: Tests if there’s a significant difference in means across Laboratory_1, Laboratory_2, Laboratory_3, and Laboratory_4.
# Display the ANOVA test results
anova_result#Stores the ANOVA test statistic and p-value.
#Problem 4: Defective Order Forms by Center
#Objective: To check if the defective percentage varies by center.
# Load the dataset
data = pd.read_csv('CustomerOrderform.csv')
print(data.head())
print(data.columns)
# Example contingency table (update this based on your data)
# Replace 'Center1', 'Center2', etc., with actual column names or values
contingency_table = pd.crosstab(data['Phillippines'], data['Indonesia'])
#Contingency Table: Creates a contingency table using data from Phillippines and Indonesia columns.
# Perform the Chi-Square Test
chi2, p, dof, expected = chi2_contingency(contingency_table)
#Chi-Square Test: Tests for independence between categories in Phillippines and Indonesia.
#chi2: Chi-square statistic.
#p: p-value for statistical significance.
#dof: Degrees of freedom.
#expected: Expected counts if no association exists.
# Display results
print(f"Chi2 Statistic: {chi2}, p-value: {p}")
#Problem 5: Differences in Store Visits by Gender on Weekdays vs. Weekends
#Objective: To determine if the percentage of males and females visiting the store differs based on the day of the week.
# Load data
data = pd.read_csv('Fantaloons.csv')
# Drop rows with NaN values
data_cleaned = data.dropna()#Removes any rows with missing values, stored as data_cleaned.
contingency_table = pd.crosstab(index=data_cleaned['Weekdays'], columns=data_cleaned['Weekend'])
#Contingency Table: Creates a cross-tabulation table for Weekdays and Weekend columns.
chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
#Chi-Square Test: Tests independence between the Weekdays and Weekend categories in cleaned data.
#chi2_stat: Chi-square test statistic.
#p_value: p-value.
#dof: Degrees of freedom.
#expected: Expected counts if no association exists.
print(f"Chi2 Statistic: {chi2_stat}, p-value: {p_value}")
# Remove rows with NaN values
fantaloons_data_cleaned = data.dropna()
# Create a contingency table for the count of Males and Females on Weekdays and Weekends
contingency_table = pd.crosstab(index=fantaloons_data_cleaned['Weekdays'], columns=fantaloons_data_cleaned['Weekend'])
#Contingency Table: Creates a table with Weekdays and Weekend for counts of Males and Females.
# Perform the Chi-Square test for independence on this contingency table
chi2_stat, p_value, dof, expected_values = chi2_contingency(contingency_table)
#Chi-Square Test: Repeats the chi-square test on this new contingency table to check if gender distribution is independent across weekdays and weekends.
# Output the chi-square test statistic, p-value, degrees of freedom, and expected values
contingency_table, chi2_stat, p_value, dof, expected_values
