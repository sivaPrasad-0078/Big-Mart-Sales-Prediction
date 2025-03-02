#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


# In[2]:


train_data = pd.read_csv(r"C:\Users\user\Downloads\siva\Hackthon\train_v9rqX0R.csv")
test_data = pd.read_csv(r"C:\Users\user\Downloads\siva\Hackthon\test_AbJTz2l.csv")


# # EDA

# In[3]:


train_data.head()


# In[4]:


print('Train Data Shape :', train_data.shape)


# In[5]:


train_data.isna().sum()


# In[6]:


train_data.info()


# In[7]:


train_data.describe()


# In[9]:


import seaborn as sns


# In[10]:


sns.boxplot(y= 'Item_Weight', data = train_data)


# In[11]:


test_data.head()


# In[12]:


print('Test Data Shape :', test_data.shape)


# In[13]:


test_data.isna().sum()


# In[14]:


test_data.info()


# In[15]:


test_data.describe()


# In[16]:


sns.boxplot(y='Item_Weight',data = test_data)


# # Imputing Null Values

# In[17]:


# Null values contained columns in train data
# Item_Weight, Outlet_Size

train_data['Item_Weight'] = train_data['Item_Weight'].fillna(train_data['Item_Weight'].mean())


# In[18]:


mode_outlet_size=train_data['Outlet_Size'].mode()[0]


# In[19]:


train_data['Outlet_Size'] = train_data['Outlet_Size'].fillna(mode_outlet_size)


# In[20]:


train_data.isna().sum()


# In[21]:


test_data['Item_Weight'] = test_data['Item_Weight'].fillna(test_data['Item_Weight'].mean())


# In[22]:


mode_outlet_size = test_data['Outlet_Size'].mode()[0]


# In[23]:


test_data['Outlet_Size'] = test_data['Outlet_Size'].fillna(mode_outlet_size)


# In[24]:


test_data.isna().sum()


# In[25]:


train_data['Item_Fat_Content'].value_counts()


# In[26]:


train_data['Item_Fat_Content'] = train_data['Item_Fat_Content'].replace({'LF':'Low Fat','low fat':'Low Fat','reg':'Regular'})


# In[27]:


train_data['Item_Fat_Content'].value_counts()


# In[28]:


test_data['Item_Fat_Content'].value_counts()


# In[29]:


test_data['Item_Fat_Content'] = test_data['Item_Fat_Content'].replace({'LF':'Low Fat','low fat':'Low Fat','reg':'Regular'})


# In[30]:


test_data['Item_Fat_Content'].value_counts()


# In[31]:


train_data['Item_Type'].value_counts()


# In[32]:


def col_classifier(x,q1,median,q3):
    if x< q1:
        return 1
    elif x>q1 and x<= median:
        return 2
    elif x>median and x<=q3:
        return 3
    elif x>q3:
        return 4


# In[33]:


q1 = train_data['Item_Visibility'].quantile(0.25)
median = train_data['Item_Visibility'].quantile(0.50)
q3 = train_data['Item_Visibility'].quantile(0.75)

train_data['Item_Visibility'] = train_data['Item_Visibility'].apply(lambda x: col_classifier(x,q1,median,q3))
test_data['Item_Visibility'] = test_data['Item_Visibility'].apply(lambda x: col_classifier(x,q1,median,q3))


# In[34]:


#It is not used for the predictions
train_data = train_data.drop(columns = ['Item_Identifier'])
test_data = test_data.drop(columns = ['Item_Identifier'])


# In[35]:


train_data['Item_Outlet_Sales'] = np.log1p(train_data['Item_Outlet_Sales'])


# In[36]:


cat_cols = ['Item_Fat_Content', 'Item_Type', 'Outlet_Identifier', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']
num_cols = ['Item_Visibility', 'Item_Weight', 'Outlet_Establishment_Year']


# In[37]:


col_trans = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
        ('num', StandardScaler(), num_cols)
    ]
)


# In[51]:


col_trans


# In[38]:


model = RandomForestRegressor(random_state=42)


# In[39]:


pipeline = Pipeline(steps=[
    ('preprocessor', col_trans),
    ('regressor', model)
])


# In[52]:


pipeline


# In[40]:


X = train_data.drop(columns=['Item_Outlet_Sales'])
y = train_data['Item_Outlet_Sales']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# In[41]:


param_dist = {
    'regressor__n_estimators': [100, 200, 300, 400, 500],
    'regressor__max_depth': [None, 10, 20, 30, 40],
    'regressor__min_samples_split': [2, 5, 10],
    'regressor__min_samples_leaf': [1, 2, 4]
}


# In[42]:


random_search = RandomizedSearchCV(
    pipeline, param_distributions=param_dist, n_iter=20, cv=3, verbose=2, n_jobs=-1, random_state=42, scoring='neg_mean_squared_error'
)


# In[43]:


random_search.fit(X_train, y_train)


# In[44]:


y_pred = random_search.best_estimator_.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))


# In[45]:


print("Best Parameters:", random_search.best_params_)
print("RMSE on validation set:", rmse)


# # Predictions for Test Data

# In[46]:


X_test = test_data
test_predictions = np.expm1(random_search.best_estimator_.predict(X_test))


# # Submission file

# In[47]:


test_data_original = pd.read_csv(r"C:\Users\user\Downloads\siva\Hackthon\test_AbJTz2l.csv")


# In[49]:


submission = pd.DataFrame({
    'Item_Identifier': test_data_original['Item_Identifier'],
    'Outlet_Identifier': test_data_original['Outlet_Identifier'],
    'Item_Outlet_Sales': test_predictions
})


# In[50]:


submission.to_csv(r"C:\Users\user\Downloads\siva\Hackthon\submission.csv",index = False)

