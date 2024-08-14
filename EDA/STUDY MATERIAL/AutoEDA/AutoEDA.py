# Load the Data
# Import the pandas library
import pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_csv(r"education.csv")


# Auto EDA
# ---------
# Sweetviz
# Autoviz
# Dtale
# Pandas Profiling
# Dataprep


# Sweetviz
###########
pip install sweetviz
# Import the sweetviz library
import sweetviz as sv

# Analyze the DataFrame and generate a report
s = sv.analyze(df)

# Display the report in HTML format
s.show_html()



# Autoviz
###########
# pip install autoviz
# Import the AutoViz_Class from the autoviz package
from autoviz.AutoViz_Class import AutoViz_Class

# Create an instance of AutoViz_Class
av = AutoViz_Class()

# Generate visualizations for the dataset
a = av.AutoViz(r"D:/Data/education.csv", chart_format='html')

# Get the current working directory
import os
os.getcwd()


# If the dependent variable is known:
a = av.AutoViz(r"C:/Data/education.csv", depVar = 'gmat') # depVar - target variable in your dataset



# D-Tale
########

# pip install dtale   # In case of any error then please install werkzeug appropriate version (pip install werkzeug==2.0.3)
import dtale
import pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_csv(r"education.csv")

# Display the DataFrame using D-Tale
d = dtale.show(df)

# Open the browser to view the interactive D-Tale dashboard
d.open_browser()



# Pandas Profiling
###################

pip install pandas_profiling
from pandas_profiling import ProfileReport 

p = ProfileReport(df)

# Display the profile report
p

# Save the profile report to an HTML file
p.to_file("output.html")

import os#importing the os module
# Get the current working directory
os.getcwd()




# dataprep
###################
from dataprep.eda import create_report
# Generate a profile report using pandas-profiling

# Generate an EDA report using dataprep
report = create_report(df, title='My Report')

# Show the EDA report in the browser
report.show_browser()



