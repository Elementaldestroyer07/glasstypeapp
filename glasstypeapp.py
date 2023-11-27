# S2.1: Open Sublime text editor, create a new Python file, copy the following code in it and save it as 'glass_type_app.py'.
# You have already created this ML model in ones of the previous classes.

# Importing the necessary Python modules.
import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score

# ML classifier Python modules
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Loading the dataset.
@st.cache()
def load_data():
    file_path = "glass-types.csv"
    df = pd.read_csv(file_path, header = None)
    # Dropping the 0th column as it contains only the serial numbers.
    df.drop(columns = 0, inplace = True)
    column_headers = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'GlassType']
    columns_dict = {}
    # Renaming columns with suitable column headers.
    for i in df.columns:
        columns_dict[i] = column_headers[i - 1]
        # Rename the columns.
        df.rename(columns_dict, axis = 1, inplace = True)
    return df

glass_df = load_data()

# Creating the features data-frame holding all the columns except the last column.
X = glass_df.iloc[:, :-1]

# Creating the target series that holds last column.
y = glass_df['GlassType']

# Spliting the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

@st.cache()
def prediction(model, RI, Na, Mg, Al, Si, K, Ca, Ba, Fe):

    pred = model.predict([RI, Na, Mg, Al, Si, K, Ca, Ba, Fe])
    pred = pred[0]
    if pred == 1:
        return 'building windows float processed'.upper()
    elif pred == 2:
        return 'building windows non float processed'.upper()
    elif pred == 3:
        return 'vehicle windows float processed'.upper()
    elif pred == 4:
        return 'vehicle windows non float processed'.upper()
    elif pred == 5:
        return 'containers'.upper()
    elif pred == 6:
        return 'tableware'
    else:
        return 'headlamps'

st.title('glass type predictor')
st.sidebar.title('exploratory data analysis')
if st.sidebar.checkbox('show raw data'):
    st.subheader('full dataset')
    st.dataframe(glass_df)
st.sidebar.subheader('scatter plot')
features_list = st.sidebar.multiselect('select the x axis value', ('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'))
st.set_option('deprecation.showPyplotGlobalUse', False)
for feature in features_list:
    st.subheader(f'scatterplot between {feature} and glasstype')
    plt.figure(figsize=(20,7))
    sns.scatterplot(x = feature, y = 'GlassType', data = glass_df)
    st.pyplot()

#make histogram

st.sidebar.subheader('histogram')
hist_feat = st.sidebar.multiselect('select the X axis value for histogram', ('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'))
for feat in hist_feat:
    st.subheader(f'histogram between {feat} and glasstype')
    plt.figure(figsize=(20,7))
    plt.hist(glass_df[feat], bins = 'sturges', color = 'black')
    st.pyplot()

#make boxplot
st.sidebar.subheader('boxplot')
box_feat = st.sidebar.multiselect('select the value', ('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'))
for bfeat in box_feat:
    st.subheader(f'boxplot for {bfeat}')
    plt.figure(figsize=(10,2))
    sns.boxplot(glass_df[bfeat])
    st.pyplot()






