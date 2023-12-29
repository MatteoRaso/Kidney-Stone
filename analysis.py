import numpy as np
import pandas as pd
import scipy.stats as st

data = pd.read_csv("kindey stone urine analysis.csv")
target_0 = data[data.target == 0]
target_1 = data[data.target == 1]
# We don't need to get the "target" column at the back.
columns = data.columns[:-1]

for column in columns:
    test_0 = st.normaltest(target_0[column])
    test_1 = st.normaltest(target_1[column])
    perform_test = True

    if test_0.pvalue < 0.05:
        print("The distribution for " + column + " is not normal at target = 0.")
        if test_0.statistic < 1:
            print("The deviation is small. Should be safe to ignore.")

        else:
            print("The deviation is non-trivial. Use a different test.")
            print(test_0.statistic)
            perform_test = False

    if test_1.pvalue < 0.05:
        print("The distribution for " + column + " is not normal at target = 1.")
        if test_1.statistic < 1:
            print("The deviation is small. Should be safe to ignore.")

        else:
            print("The deviation is non-trivial. Use a different test.")
            print(test_1.statistic)
            perform_test = False

    if perform_test:
        result = st.ttest_ind(target_0[column], target_1[column], equal_var=False)
        if result.pvalue < 0.05:
            print(column + " is a significant variable.")


