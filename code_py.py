#!/usr/bin/env python
# coding: utf-8

# In[25]:


#!pip install imbalanced-learn


# In[26]:



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split


from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import recall_score, confusion_matrix, precision_score, f1_score, accuracy_score, classification_report
import sklearn.metrics as metrics


# In[27]:


#Preparation and cleaning of the data
df = pd.read_csv('train.csv', sep=';')#reading dataset, test was a part of the train dataset, that is why we used only the train dataset
df.info()


# In[28]:


df.y.value_counts()


# In[29]:


df.columns = df.columns.str.replace(' ', '') #remove white spaces
df = df.rename({'y': 'target'}, axis=1)    #rename the 'y' to 'target' 
print(df.head())  #check the renaming


# In[30]:


print(df.duplicated().value_counts())  #check duplicated values


# In[31]:


#Data Quality Report for numeric features
#for train dataset
dqr_cont = df.describe()
cardinality = df.apply(pd.Series.nunique)
dqr_cont.loc['cardinality'] = cardinality[dqr_cont.columns]
dqr_cont.loc['missing'] = df.isnull().sum(axis = 0)
dqr_cont = dqr_cont.T

print(dqr_cont)


# In[32]:


categorical_columns=["job","marital","education","default","housing","loan","contact","month","poutcome","target"]
formatter = "{0:.2f}"
listModeName=[]
list_mode=[]
print(f"{'Feature' : <10}{'Count' : ^10}{'Miss' : ^10}{'Card' : ^10}{'Mode':^10}{'Mode Freq':^15}{'Mode(%)':^15}{'2nd Mode':^15}{'2nd Mode Freq':^15}{'2nd Mode(%)':^15}")
for i in range(len(categorical_columns)):
    listModeName.extend(df[categorical_columns[i]].value_counts().index.tolist())  
    list_mode.extend(df[categorical_columns[i]].value_counts())  
    print(f"{categorical_columns[i].upper(): <10}{df[categorical_columns[i]].count(): ^10}{df[categorical_columns[i]].isnull().sum(): ^10}{df[categorical_columns[i]].nunique(): ^10}{df[categorical_columns[i]].mode().values[0]: ^10}{df[categorical_columns[i]].value_counts().max(): ^15}{formatter.format((df[categorical_columns[i]].value_counts().max()/df[categorical_columns[i]].count())*100): ^15}{listModeName[1]: ^15}{list_mode[1]: ^15}{formatter.format((list_mode[1]/df[categorical_columns[i]].count())*100): ^15}")
    list_mode.clear() 
    listModeName.clear()


# In[33]:


out=[]

def iqr_outliers(df):
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    iqr = q3-q1
    Lower_tail = q1 - 1.5 * iqr
    Upper_tail = q3 + 1.5 * iqr
    for i in df:
        if i > Upper_tail or i < Lower_tail:
            out.append(i)
    print("Outliers:", len(out))
          
iqr_outliers(df['previous'])


# In[34]:


#Data Visulation
df.groupby('job').size().plot(kind='pie', autopct='')
plt.title('Piechart for job')
plt.show()


print(df["education"].value_counts())
df.groupby('education').size().plot(kind='pie', autopct='%.2f')
plt.title('Piechart for education')
plt.show()

print(df["marital"].value_counts())
df.groupby('marital').size().plot(kind='pie', autopct='%.2f')
plt.title('Piechart for marital')
plt.show()

print(df["housing"].value_counts())
df.groupby('housing').size().plot(kind='pie', autopct='%.2f')
plt.title('Piechart for housing')
plt.show()

print(df["default"].value_counts())
df.groupby('default').size().plot(kind='pie', autopct='%.2f')
plt.title('Piechart for default')
plt.show()

print(df["loan"].value_counts())
df.groupby('loan').size().plot(kind='pie', autopct='%.2f')
plt.title('Piechart for loan')
plt.show()

print(df["contact"].value_counts())
df.groupby('contact').size().plot(kind='pie', autopct='%.2f')
plt.title('Piechart for contact')
plt.show()

print(df["poutcome"].value_counts())
df.groupby('poutcome').size().plot(kind='pie', autopct='%.2f')
plt.title('Piechart for poutcome')
plt.show()

print(df["target"].value_counts())
df.groupby('target').size().plot(kind='pie', autopct='%.2f')
plt.title('Piechart for target')
plt.show()

sns.set_theme(style='darkgrid')
sns.set(rc = {'figure.figsize':(7, 7)})
target = sns.countplot(x="target", data = df, order = df["target"].value_counts().index)
target.tick_params(axis='x', rotation=60)
plt.title("Univariate analysis of the target")
plt.show()

sns.set_theme(style='darkgrid')
sns.set(rc = {'figure.figsize':(7, 7)})
marital = sns.countplot(x="job", data = df, hue = "target", order = df["job"].value_counts().index)
marital.tick_params(axis='x', rotation=300)
plt.title("Analysis of the relationship between jobs and target feature")
plt.show()

sns.set_theme(style='darkgrid')
sns.set(rc = {'figure.figsize':(7, 7)})
marital = sns.countplot(x="education", data = df, hue = "target", order = df["education"].value_counts().index)
marital.tick_params(axis='x')
plt.title("Analysis of the relationship between education and target feature")
plt.show()

sns.set_theme(style='darkgrid')
sns.set(rc = {'figure.figsize':(7, 7)})
marital = sns.countplot(x="marital", data = df, hue = "target", order = df["marital"].value_counts().index)
marital.tick_params(axis='x')
plt.title("Analysis of the relationship between marital and target feature")
plt.show()

sns.set_theme(style='darkgrid')
sns.set(rc = {'figure.figsize':(7, 7)})
sns.countplot(data = df, x="loan", hue ="target")
plt.title("Analysis of the relationship between loan and target feature")
plt.show()

sns.set_theme(style='darkgrid')
sns.set(rc = {'figure.figsize':(7, 7)})
sns.countplot(x="housing", data = df, hue ="target")
plt.title("Analysis of the relationship between housing and target feature")
plt.show()

sns.set_theme(style='darkgrid')
sns.set(rc = {'figure.figsize':(7, 7)})
sns.countplot(x="contact", data = df, hue ="target", order = df["contact"].value_counts().index)
plt.title("Analysis of the relationship between contact and target feature")
plt.show()

sns.set_theme(style='darkgrid')
sns.set(rc = {'figure.figsize':(7, 7)})
month = sns.countplot(x="month", data = df, hue = "target", order = df["month"].value_counts().index)
month.tick_params(axis='x', rotation=60)
plt.title("Analysis of the relationship between month and the target feature")
plt.show()

sns.set_theme(style='darkgrid')
sns.set(rc = {'figure.figsize':(7, 7)})
poutcome = sns.countplot(x="poutcome", data = df, hue = "target", order = df["poutcome"].value_counts().index)
poutcome.tick_params(axis='x', rotation=60)
plt.title("Analysis of the relationship between poutcome and the target feature")
plt.show()

#histograms
plt.figure(figsize=(16,5))
plt.subplot(1,2,1)
plt.hist(df['age'], color = 'violet', edgecolor = 'black',
         bins = int(180/10))

# seaborn histogram
sns.distplot(df['age'], hist=True, kde=False, 
             bins=int(180/10), color = 'violet',
             hist_kws={'edgecolor':'black'})
# Add labels
plt.title('Histogram of Age')
plt.xlabel('Age')
plt.ylabel('Count')

plt.subplot(1,2,2)
plt.hist(df['balance'], color = 'violet', edgecolor = 'black',
         bins = int(180/5))

# seaborn histogram
sns.distplot(df['balance'], hist=True, kde=False, 
             bins=int(180/5), color = 'violet',
             hist_kws={'edgecolor':'black'})
# Add labels
plt.title('Histogram of Balance')
plt.xlabel('Balance')
plt.ylabel('Count')

plt.figure(figsize=(16,5))
plt.subplot(1,2,1)
plt.hist(df['day'], color = 'violet', edgecolor = 'black',
         bins = int(180/10))

# seaborn histogram
sns.distplot(df['day'], hist=True, kde=False, 
             bins=int(180/10), color = 'violet',
             hist_kws={'edgecolor':'black'})
# Add labels
plt.title('Histogram of Day')
plt.xlabel('Day')
plt.ylabel('Count')

plt.subplot(1,2,2)
plt.hist(df['duration'], color = 'violet', edgecolor = 'black',
         bins = int(180/5))

# seaborn histogram
sns.distplot(df['duration'], hist=True, kde=False, 
             bins=int(180/5), color = 'violet',
             hist_kws={'edgecolor':'black'})
# Add labels
plt.title('Histogram of Duration')
plt.xlabel('Duration')
plt.ylabel('Count')

plt.figure(figsize=(16,5))
plt.subplot(1,2,1)
plt.hist(df['campaign'], color = 'violet', edgecolor = 'black',
         bins = int(180/10))

# seaborn histogram
sns.distplot(df['campaign'], hist=True, kde=False, 
             bins=int(180/10), color = 'violet',
             hist_kws={'edgecolor':'black'})
# Add labels
plt.title('Histogram of Campaign')
plt.xlabel('Campaign')
plt.ylabel('Count')

plt.subplot(1,2,2)
plt.hist(df['pdays'], color = 'violet', edgecolor = 'black',
         bins = int(180/5))

# seaborn histogram

sns.distplot(df['pdays'], hist=True, kde=False, 
             bins=int(180/5), color = 'violet',
             hist_kws={'edgecolor':'black'})
# Add labels
plt.title('Histogram of Pdays')
plt.xlabel('Pdays')
plt.ylabel('Count')

plt.figure(figsize=(16,5))
plt.subplot(1,2,1)
plt.hist(df['previous'], color = 'violet', edgecolor = 'black',
         bins = 18)

# Add labels
plt.title('Histogram of Previous')
plt.xlabel('Previous')
plt.ylabel('Count')

warnings.filterwarnings('ignore')
plt.figure(figsize=(16,5))
plt.subplot(1,2,1)
sns.distplot(df.age)
plt.subplot(1,2,2)
sns.distplot(df.balance)

plt.figure(figsize=(16,5))
plt.subplot(1,2,1)
sns.distplot(df.day)
plt.subplot(1,2,2)
sns.distplot(df.duration)

plt.figure(figsize=(16,5))
plt.subplot(1,2,1)
sns.distplot(df.campaign)
plt.subplot(1,2,2)
sns.distplot(df.pdays)

plt.figure(figsize=(16,5))
sns.distplot(df.previous)

plt.figure(figsize=(16,5))
plt.subplot(1,2,1)
sns.boxplot(df['age'])

#First quartile (Q1)
Q1 = np.percentile(df['age'], 25, interpolation = 'midpoint')
print("Q1 of age: ",Q1)  
# Third quartile (Q3)
Q3 = np.percentile(df['age'], 75, interpolation = 'midpoint')
print("Q3 of age: ",Q3)   
# Interquaritle range (IQR)
IQR = Q3 - Q1 
print("IQR of age: ",IQR)
plt.subplot(1,2,2)
sns.boxplot(df['balance'])
Q1 = np.percentile(df['balance'], 25, interpolation = 'midpoint')
print("Q1 of balance: ",Q1)  
# Third quartile (Q3)
Q3 = np.percentile(df['balance'], 75, interpolation = 'midpoint')
print("Q3 of balance: ",Q3)   
# Interquaritle range (IQR)
IQR = Q3 - Q1 
print("IQR of balance: ",IQR)
print("Count of age feature:", df['age'].count())


plt.figure(figsize=(16,5))
plt.subplot(1,2,1)
sns.boxplot(df['day'])
plt.subplot(1,2,2)
sns.boxplot(df['duration'])

#First quartile (Q1)
Q1 = np.percentile(df['day'], 25, interpolation = 'midpoint')
print("Q1 of day: ",Q1)  
# Third quartile (Q3)
Q3 = np.percentile(df['day'], 75, interpolation = 'midpoint')
print("Q3 of day: ",Q3)   
# Interquaritle range (IQR)
IQR = Q3 - Q1 
print("IQR of day: ",IQR)

Q1 = np.percentile(df['duration'], 25, interpolation = 'midpoint')
print("Q1 of duration: ",Q1)  
# Third quartile (Q3)
Q3 = np.percentile(df['duration'], 75, interpolation = 'midpoint')
print("Q3 of duration: ",Q3)   
# Interquaritle range (IQR)
IQR = Q3 - Q1 
print("IQR of duration: ",IQR)


plt.figure(figsize=(16,5))
plt.subplot(1,2,1)
sns.boxplot(df['campaign'])
plt.subplot(1,2,2)
sns.boxplot(df['pdays'])

#First quartile (Q1)
Q1 = df['campaign'].quantile(0.25)
print("Q1 of campaign2: ",Q1) 
# Third quartile (Q3)
Q3 = df['campaign'].quantile(0.75)
print("Q3 of campaign2: ",Q3)
# Interquaritle range (IQR)
IQR = Q3 - Q1
print("IQR of campaign: ",IQR)

Q1 = np.percentile(df['pdays'], 25, interpolation = 'midpoint')
print("Q1 of pdays: ",Q1)  
# Third quartile (Q3)
Q3 = np.percentile(df['pdays'], 75, interpolation = 'midpoint')
print("Q3 of pdays: ",Q3)   
# Interquaritle range (IQR)
IQR = Q3 - Q1 
print("IQR of pdays: ",IQR)


plt.figure(figsize=(16,5))
plt.subplot(1,2,1)
sns.boxplot(df['campaign'])
plt.subplot(1,2,2)
sns.boxplot(df['pdays'])


plt.figure(figsize=(16,5))
sns.boxplot(df['previous'])
Q1 = np.percentile(df['previous'], 25, interpolation = 'midpoint')
print("Q1 of previous: ",Q1)  
# Third quartile (Q3)
Q3 = np.percentile(df['previous'], 75, interpolation = 'midpoint')
print("Q3 of previous: ",Q3)   
# Interquaritle range (IQR)
IQR = Q3 - Q1 
print("IQR of previous: ",IQR)


plt.figure(figsize=(16,5))
sns.boxplot(df['previous'])
Q1 = np.percentile(df['previous'], 25, interpolation = 'midpoint')
print("Q1 of previous: ",Q1)  
# Third quartile (Q3)
Q3 = np.percentile(df['previous'], 75, interpolation = 'midpoint')
print("Q3 of previous: ",Q3)   
# Interquaritle range (IQR)
IQR = Q3 - Q1 
print("IQR of previous: ",IQR)


# In[35]:


num_cols = ['balance', 'duration', 'campaign', 'pdays', 'previous']
for i in num_cols:
    plt.figure(i)
    sns.scatterplot(data=df, x="age", y=i)
    


# In[36]:


df= df[df['duration'] < 4000]  #outlier handling
df= df[df['previous'] < 100]
df= df[df['campaign'] < 50]
df= df[df['pdays'] < 600]
df= df[df['balance'] < 80000]


# In[37]:


df.info()                     


# In[38]:


df.job.value_counts()


# In[39]:


#get ratio between job and education
education_ratio = pd.DataFrame({'Job' : []})
for i in df["job"].unique():
    education_ratio = education_ratio.append(df[(df["job"] == i)]["education"].value_counts().to_frame().iloc[0] * 100 / df[(df["job"] == i)]["education"].value_counts().sum())
education_ratio["Job"] = df["job"].unique()

print(education_ratio)


# In[40]:


#replace the unknown values
df.loc[(df.job == "unknown") & (df.education == "secondary"),"job"] = "services"
df.loc[(df.job == "unknown") & (df.education == "primary"),"job"] = "housemaid"
df.loc[(df.job == "unknown") & (df.education == "tertiary"),"job"] = "management"
df.loc[(df.job == "unknown"),"job"] = "blue-collar"

df.loc[(df.education == "unknown") & (df.job == "admin."),"education"] = "secondary"
df.loc[(df.education == "unknown") & (df.job == "management"),"education"] = "secondary"
df.loc[(df.education == "unknown") & (df.job == "services"),"education"] = "tertiary"
df.loc[(df.education == "unknown") & (df.job == "technician."),"education"] = "secondary"
df.loc[(df.education == "unknown") & (df.job == "retired"),"education"] = "secondary"
df.loc[(df.education == "unknown") & (df.job == "blue-collar"),"education"] = "secondary"
df.loc[(df.education == "unknown") & (df.job == "housemaid."),"education"] = "primary"
df.loc[(df.education == "unknown") & (df.job == "self-employed"),"education"] = "tertiary"
df.loc[(df.education == "unknown") & (df.job == "student"),"education"] = "secondary"
df.loc[(df.education == "unknown") & (df.job == "entrepreneur"),"education"] = "tertiary"
df.loc[(df.education == "unknown") & (df.job == "unemployed"),"education"] = "secondary"

df.loc[(df.education == "unknown"),"education"] = "secondary"

df["contact"].replace(["unknown"],df["contact"].mode(),inplace = True)


# In[41]:


df.education.value_counts()
df.marital.value_counts()


# In[42]:


df = pd.get_dummies(df, columns=['job', 'marital', 'education', 'poutcome'])


# In[43]:


df.info()


# In[44]:


#encoding the features that have 2 values
binary_cols = ['housing', 'default', 'loan', 'contact', 'target'] 

for i in binary_cols: 
    df[i].replace({'no': 0, 'yes': 1, 'telephone': 0, 'cellular': 1}, inplace=True) 

#df.info()


# In[45]:


#drop unrelated feautes
df.drop(columns = ["day","month"],inplace = True) 


# In[46]:


scaler = MinMaxScaler(feature_range=(0, 1), copy=True)  #max min normalization [0,1]
scaler.fit(df)
df = pd.DataFrame(scaler.transform(df), index=df.index, columns=df.columns)
print(df.info())


# In[47]:


plt.figure(figsize=(15,8))
df.corr()['target'].sort_values(ascending = False).plot(kind='bar') #Correlation of "target" with other features:


# In[48]:


y = df.target.to_frame()  #25/75 testing
X = df.drop(columns = ["target"])
X_train , X_test , y_train , y_test = train_test_split(X,y, test_size = 0.25, random_state = 10)


# In[49]:


models = [LogisticRegression(solver='liblinear'), #models to be trained
          KNeighborsClassifier(),
          SGDClassifier(),
          DecisionTreeClassifier(),
          RandomForestClassifier(),
          SVC()
        ]


# In[50]:


for i, model in enumerate(models):        #25/75 and 10-fold training
    model.fit(X_train, y_train)
    print(models[i], ':', model.score(X_test, y_test))
    y_pred = model.predict(X_test)
    conf_mat = confusion_matrix(y_test, y_pred)
    #acc = accuracy_score(y_test, y_pred)
    print("Confusion Matrix : ")
    print(conf_mat)
    print(classification_report(y_test, y_pred))
    print('Cross Validation mean:',(cross_val_score(model, X_train, y_train, cv=10, n_jobs=2, scoring = 'accuracy').mean()))
    print()


# In[51]:


sm = SMOTE() #oversampling
X_sm , y_sm = sm.fit_resample(X, y)
y_sm.target.value_counts()


# In[ ]:


X_train_sm , X_test_sm , y_train_sm , y_test_sm = train_test_split(X_sm, y_sm, test_size = 0.25, random_state = 10)
  #same process with before on the oversampled dataset
for i, model in enumerate(models):
    model.fit(X_train_sm, y_train_sm)
    print(models[i], ':', model.score(X_test_sm, y_test_sm))
    y_pred_sm = model.predict(X_test_sm)
    conf_mat = confusion_matrix(y_test_sm, y_pred_sm)
    #acc = accuracy_score(y_test, y_pred)
    print("Confusion Matrix : ")
    print(conf_mat)
    print(classification_report(y_test_sm, y_pred_sm))
    print('Cross Validation mean:',(cross_val_score(model, X_train_sm, y_train_sm, cv=10, n_jobs=2, scoring = 'accuracy').mean()))
    print()
    from sklearn.metrics import r2_score
    r2score = r2_score(y_test_sm, y_pred_sm)
    print('R2 Score: ', r2score)
    from sklearn.metrics import mean_absolute_error
    mae = mean_absolute_error(y_test_sm, y_pred_sm)
    print('Mean Absolute Error: ', mae)
    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(y_test_sm, y_pred_sm)
    print('Mean Squared Error' ,mse)
    from sklearn.metrics import roc_curve, roc_auc_score
    fpr, tpr, thresholds = roc_curve(y_test_sm, y_pred_sm)
    sns.set()

    plt.plot(fpr, tpr)

    plt.plot(fpr, fpr, linestyle = '--', color = 'k')

    plt.xlabel('False positive rate')

    plt.ylabel('True positive rate')

    AUROC = np.round(roc_auc_score(y_test_sm, y_pred_sm), 2)

    plt.title(f'ROC curve; AUROC: {AUROC}');

    plt.show()


# In[ ]:





# In[ ]:




