import pandas as pd
df = pd.read_csv(r"Cars.csv")
import sweetviz as sv
s=sv.analyze(df)
s.show_html()


from autoviz.AutoViz_Class import AutoViz_Class
av = AutoViz_Class()
a = av.AutoViz(r"Cars.csv", chart_format='html')
import os
os.getcwd()
a = av.AutoViz(r"Cars.csv", depVar = 'WT') 

import dtale
import pandas as pd
df = pd.read_csv(r"education.csv")
d = dtale.show(df)
d.open_browser()

pip install pandas_profiling
pip install ydata-profiling
from ydata_profiling import ProfileReport
from pandas_profiling import ProfileReport
p = ProfileReport(df)
p
p.to_file("output.html")
import os
os.getcwd()

pip install dataprep
from dataprep.eda import create_report
report = create_report(df, title='My Report')
report.show_browser()