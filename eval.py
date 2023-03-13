import os
import pickle
import pandas as pd
from metrics import plot_loss, plot_accuracy, plot_learning_rate

# specify model name to evaluate
model_name = 'NiN'

# get current working directory
mo = 'models'
cwd = os.getcwd()
model_dir = os.path.join(cwd, mo)
model_path = os.path.join(model_dir, model_name + ".pt")
log_dict_path = os.path.join(model_dir, model_name + "_log_dict.pkl")
# open log dict from pickle
with open(log_dict_path, "rb") as f:
    log_dict = pickle.load(f)

### Evaluate the model
plot_loss(log_dict)
plot_accuracy(log_dict)
plot_learning_rate(log_dict)

# remove empty lists from log_dict
# avoid this error: dictionary changed size during iteration
log_dict = {k: v for k, v in log_dict.items() if v}

# store log dict as dataframe
df = pd.DataFrame.from_dict(log_dict)
# remove columns with all NaN values
df = df.dropna(axis=1, how='all')
# inspect dataframe
# add another column with epoch number, same as index, as first column
df.insert(0, 'epoch', df.index)
pd.set_option('display.max_columns', None)
print(df.head())