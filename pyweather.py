
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pickle
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_score

warnings.filterwarnings('ignore')

pd.set_option("display.max_columns",None)

dataset = 'weatherAUS.csv'

rain = pd.read_csv(dataset)

rain = rain.replace(np.inf, np.nan)

print(rain.head())

print(rain.shape)

print(rain.info())

print(rain.describe(exclude=[object]))

print(rain.describe(include=[object]))

categorical_features = [column_name for column_name in rain.columns if rain[column_name].dtype == 'O']
print("Number of Categorical Features: {}".format(len(categorical_features)))
print("Categorical Features: ",categorical_features)

numerical_features = [column_name for column_name in rain.columns if rain[column_name].dtype != 'O']
print("Number of Numerical Features: {}".format(len(numerical_features)))
print("Numerical Features: ",numerical_features)

for each_feature in categorical_features:
    print("Cardinality(no. of unique values) of {} are: {}".format(each_feature,len(rain[each_feature].unique())))

rain['Date'] = pd.to_datetime(rain['Date'])
print(rain['Date'].dtype)
rain['year'] = rain['Date'].dt.year
rain['month'] = rain['Date'].dt.month
rain['day'] = rain['Date'].dt.day

rain.drop('Date', axis = 1, inplace = True)
print(rain.head())

categorical_features = [column_name for column_name in rain.columns if rain[column_name].dtype == 'O']
print("Number of Categorical Features: {}".format(len(categorical_features)))
print("Categorical Features: ",categorical_features)

numerical_features = [column_name for column_name in rain.columns if rain[column_name].dtype != 'O']
print("Number of Numerical Features: {}".format(len(numerical_features)))
print("Numerical Features: ",numerical_features)

print(rain[categorical_features].isnull().sum())


categorical_features_with_null = [feature for feature in categorical_features if rain[feature].isnull().sum()]

for each_feature in categorical_features_with_null:
    mode_val = rain[each_feature].mode()[0]
    rain[each_feature].fillna(mode_val,inplace=True)

print(rain[categorical_features].isnull().sum())


rain[numerical_features].isnull().sum()

'''plt.figure(figsize=(15,10))
sns.heatmap(rain[numerical_features].isnull(),linecolor='white')
plt.show()

rain[numerical_features].isnull().sum().sort_values(ascending = False).plot(kind = 'bar')
plt.show()
'''

'''for feature in numerical_features:
    plt.figure(figsize=(10,10))
    sns.boxplot(rain[feature])
    plt.title(feature)
    plt.show()
'''

print(rain[numerical_features].describe())

features_with_outliers = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'WindGustSpeed','WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Pressure9am', 'Pressure3pm', 'Temp9am', 'Temp3pm']
for feature in features_with_outliers:
    q1 = rain[feature].quantile(0.25)
    q3 = rain[feature].quantile(0.75)
    IQR = q3-q1
    lower_limit = q1 - (IQR*1.5)
    upper_limit = q3 + (IQR*1.5)
    rain.loc[rain[feature]<lower_limit,feature] = lower_limit
    rain.loc[rain[feature]>upper_limit,feature] = upper_limit

'''for feature in numerical_features:
    plt.figure(figsize=(10,10))
    sns.boxplot(rain[feature])
    plt.title(feature)
    plt.show()
'''

numerical_features_with_null = [feature for feature in numerical_features if rain[feature].isnull().sum()]
print(numerical_features_with_null)

for feature in numerical_features_with_null:
    mean_value = rain[feature].mean()
    rain[feature].fillna(mean_value,inplace=True)

print(rain.isnull().sum())

print(rain.head())
      
'''rain['RainTomorrow'].value_counts().plot(kind='bar')
plt.show()

sns.countplot(data=rain, x="RainToday")
plt.grid(linewidth = 0.5)
plt.show()

plt.figure(figsize=(20,10))
ax = sns.countplot(x="Location", hue="RainTomorrow", data=rain)
plt.show()

sns.lineplot(data=rain,x='Sunshine',y='Rainfall',color='goldenrod')
plt.show()
'''

num_features = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm']
'''
print(rain[num_features].hist(bins=10,figsize=(20,20)))
plt.show()
'''

print(rain[num_features].corr())

'''
plt.figure(figsize=(20,20))
sns.heatmap(rain[num_features].corr(),linewidths=0.5,annot=True,fmt=".2f")
plt.show()
'''

print(categorical_features)

rain['RainToday'].replace({'No':0, 'Yes': 1}, inplace = True)
rain['RainTomorrow'].replace({'No':0, 'Yes': 1}, inplace = True)

def encode_data(feature_name):
    mapping_dict = {}
    unique_values = list(rain[feature_name].unique())
    for idx in range(len(unique_values)):
        mapping_dict[unique_values[idx]] = idx
    print(mapping_dict)
    return mapping_dict

rain['WindGustDir'].replace(encode_data('WindGustDir'),inplace = True)
rain['WindDir9am'].replace(encode_data('WindDir9am'),inplace = True)
rain['WindDir3pm'].replace(encode_data('WindDir3pm'),inplace = True)
rain['Location'].replace(encode_data('Location'), inplace = True)

print(rain.head())

X = rain.drop(['RainTomorrow'],axis=1)
y = rain['RainTomorrow']

etr_model = ExtraTreesRegressor()
etr_model.fit(X,y)

print(etr_model.feature_importances_)

feature_imp = pd.Series(etr_model.feature_importances_,index=X.columns)
feature_imp.nlargest(10).plot(kind='barh')
plt.show()

print(feature_imp)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)
print("Length of Training Data: {}".format(len(X_train)))
print("Length of Testing Data: {}".format(len(X_test)))

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

start_time = time.time()
classifier_logreg = LogisticRegression(solver='liblinear', random_state=0)
classifier_logreg.fit(X_train, y_train)
end_time = time.time()

print("Time Taken to train: {}".format(end_time - start_time))

y_pred = classifier_logreg.predict(X_test)
print("Accuracy Score: {}".format(accuracy_score(y_test,y_pred)))

print("Train Data Score: {}".format(classifier_logreg.score(X_train, y_train)))
print("Test Data Score: {}".format(classifier_logreg.score(X_test, y_test)))


print("Confusion Matrix:")
print("\n",confusion_matrix(y_test,y_pred))

print("classification_report:")
print("\n",classification_report(y_test,y_pred))

y_pred_logreg_proba = classifier_logreg.predict_proba(X_test)

fpr, tpr, thresholds = roc_curve(y_test, y_pred_logreg_proba[:,1])

plt.figure(figsize=(6,4))
plt.plot(fpr,tpr,'-g',linewidth=1)
plt.plot([0,1], [0,1], 'k--' )
plt.title('ROC curve for Logistic Regression Model')
plt.xlabel("False Positive Rate")
plt.ylabel('True Positive Rate')
plt.show()

from sklearn.metrics import roc_auc_score
print('ROC AUC Scores: {}'.format(roc_auc_score(y_test, y_pred)))

scores = cross_val_score(classifier_logreg, X_train, y_train, cv = 5, scoring='accuracy')
print('Cross-validation scores:{}'.format(scores))

print('Average cross-validation score: {}'.format(scores.mean()))