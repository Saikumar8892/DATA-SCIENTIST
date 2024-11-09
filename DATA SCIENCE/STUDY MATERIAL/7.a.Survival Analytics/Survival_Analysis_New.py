# pip install lifelines
# Install the lifelines library for survival analysis using pip.

import pandas as pd
# Import the pandas library for data manipulation.

# Loading the survival unemployment data from a CSV file
survival_unemp = pd.read_csv(r"survival_unemployment.csv")
# Read the CSV file into a pandas DataFrame.
# 'r"C:\Users\survival_unemployment.csv"' specifies the file path.

survival_unemp.head()
# Display the first few rows of the DataFrame.

survival_unemp.describe()
# Display summary statistics of the DataFrame.

survival_unemp["spell"].describe()
# Display summary statistics of the 'spell' column, which refers to time.

# Spell is referring to time 
T = survival_unemp.spell
# Assign the 'spell' column to the variable 'T', representing survival time.

# Importing the KaplanMeierFitter model to fit the survival analysis
from lifelines import KaplanMeierFitter
# Import the KaplanMeierFitter class from the lifelines library.

# Initiating the KaplanMeierFitter model
kmf = KaplanMeierFitter()
# Create an instance of the KaplanMeierFitter model.

# Fitting KaplanMeierFitter model on Time and Events for death 
kmf.fit(T, event_observed=survival_unemp.event)
# Fit the KaplanMeierFitter model to the survival time 'T' and event indicator 'event'.
# 'event_observed=survival_unemp.event' specifies whether the event occurred or not.

# Time-line estimations plot 
kmf.plot()
# Plot the Kaplan-Meier survival curve.

# Over Multiple groups 
# For each group, here the group is 'ui'
survival_unemp.ui.value_counts()
# Count the occurrences of each unique value in the 'ui' column.

# Applying KaplanMeierFitter model on Time and Events for the group "1"
kmf.fit(T[survival_unemp.ui == 1], survival_unemp.event[survival_unemp.ui == 1], label='1')
# Fit the KaplanMeierFitter model separately for the group where 'ui' is 1.
# 'label='1'' sets the label for this group to '1'.
ax = kmf.plot()

# Applying KaplanMeierFitter model on Time and Events for the group "0"
kmf.fit(T[survival_unemp.ui == 0], survival_unemp.event[survival_unemp.ui == 0], label='0')
# Fit the KaplanMeierFitter model separately for the group where 'ui' is 0.
# 'label='0'' sets the label for this group to '0'.
kmf.plot(ax=ax)
# Plot the Kaplan-Meier survival curve for both groups on the same axis.
