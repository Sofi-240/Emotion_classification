import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

matplotlib.use("Qt5Agg")
sns.set_style(
    "whitegrid", rc={"axes.spines.right": False, "axes.spines.top": False}
)

sns.set_context(
    "paper", font_scale=3,
    rc={"axes.labelsize": 25, "xtick.labelsize": 25, "ytick.labelsize": 25, 'axes.formatter.min_exponent': 1}
)

