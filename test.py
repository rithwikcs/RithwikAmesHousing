import pandas as pd #import pandas library
import seaborn as sns #import seaborn library used for data visualization and plotting
import matplotlib.pyplot as plt #import matplotlib library that seaborn builds on for plotting
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split #import train_test_split function from sklearn library to split data into training and testing sets
from sklearn.metrics import mean_squared_error, r2_score #import mean_squared_error and r2_score from sklearn library to evaluate the model performance
import joblib

df = pd.read_csv("/Users/rithwik.chandrasekaran/AmesHousingMarket/AmesHousing.csv") #read the csv file into a pandas dataframe
''' print(df.head()) #this will print the first 5 rows of the dataframe
print(df.info()) #this will print the information about the dataframe
print(df.shape) ''' #this will print the shape of the dataframe

'''
print(df.describe()) #this will print the summary statistics of the dataframe
'''

plt.figure(figsize=(14,10)) #new plot canvas 14 inches wide and 10 inches tall
correlation_matrix = df.corr(numeric_only=True) #calculate correlations between numeric columns in df
sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False)
# correlation matrix is being plotted, cmap sets colorscheme (blue negative, red positive, white neutral)
# annot=False doesn't write numbers on each square
plt.title('Correlation Heatmap of Ames Housing Dataset') #title for the plot
plt.show() #displays the heatmap plot 



# df.isnull().sum().sort_values(ascending=False).head(20) #this will print the top 20 columns with the most null values

# df['Lot Frontage'] = df['Lot Frontage'].fillna(df['Lot Frontage'].median()) #fill missing values in Lot Frontage with median

df = df.drop('Lot Frontage', axis = 1) #drops the Lot Frontage column from df
print(df.isnull().sum()) #this will print the number of null values in each column

''' sns.boxplot(x=df['SalePrice']) # boxplot of Sale Price
plt.title('Sale Price Distribution With Outliers') #title for the plot
plt.show() #displays the boxplot of Sale Price by Overall Quality '''

numeric_cols = df.select_dtypes(include='number').columns # selects only numeric columns from df

for col in numeric_cols: # iterates through each column in numeric_cols
    Q1 = df[col].quantile(0.25) # calculates the first quartile of column
    Q3 = df[col].quantile(0.75) # calculates the third quartile of column
    IQR = Q3 - Q1 # calculates IQR
    lower_bound = Q1 - 1.5 * IQR # calculates lower bound for outliers
    upper_bound = Q3 + 1.5 * IQR # calculates upper bound for outliers
    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)] #filters outliers from columns

# num_cols = len(numeric_cols) 
# num_rows = (num_cols + 2) // 3 # calculates number of rows needed for subplots, 3 columns per row

# plt.figure(figsize=(16, 5 * num_rows)) # 5 in tall per row 

''' sns.boxplot(x='Overall Qual', y='SalePrice', data=df) #data = df ensures it reads df for dataframe
plt.title('Sale Price by Overall Quality')
plt.show() #displays the boxplot of Sale Price by Overall Quality '''

''' for i, col in enumerate(numeric_cols, 1): #enumerate means it will give us the index and the column name
    plt.subplot(num_rows, 3, i) #creates a subplot for each numeric column (3 columns)
    df[col].hist(bins=30, edgecolor='black') #histogram for each column, 30 bars
    plt.title(col) #title for each subplot, based on column name
    plt.xlabel(col)
    plt.ylabel('Frequency') 
plt.tight_layout() #adjusts layout to prevent overlap
plt.show() '''
    

    
    

''' plt.figure(figsize=(14,10)) #new plot canvas 14 inches wide and 10 inches tall
correlation_matrix = df.corr(numeric_only=True) #calculate correlations between numeric columns in df
sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False)
# correlation matrix is being plotted, cmap sets colorscheme (blue negative, red positive, white neutral)
# annot=False doesn't write numbers on each square
plt.title('Correlation Heatmap of Ames Housing Dataset') #title for the plot
plt.show() #displays the heatmap plot '''

#creating features

df['TotalSQFT'] = df['Total Bsmt SF'] + df['1st Flr SF'] + df['2nd Flr SF'] #creates a new column TotalSQFT by adding three columns together
df['House Age'] = 2023 - df['Year Built'] 
df['IsRemodeled'] = (df['Year Remod/Add'] > df['Year Built']).astype(int) #1 if remod, 0 if not
df['RemodAge'] = (df['Year Remod/Add'] - df['Year Built']) * df['IsRemodeled'] #calculates the age of the remodel, makes 0 if not remodeled
df['TotalBath'] = df['Full Bath'] + df['Half Bath'] * 0.5 + df['Bsmt Full Bath'] + df['Bsmt Half Bath'] * 0.5 #creates a new column TotalBath by adding Full Bath, Half Bath, Bsmt Full Bath, and Bsmt Half Bath together
df['TotalPorchSF'] = df['Open Porch SF'] + df['Enclosed Porch'] + df['3Ssn Porch'] + df['Screen Porch'] #creates a new column TotalPorchSF by adding Open Porch SF, Enclosed Porch, 3Ssn Porch, and Screen Porch together

new_features = ['TotalSQFT', 'House Age', 'Garage Area', 'TotalBath'] #list of new features created
correlations = df[new_features + ['SalePrice']].corr()['SalePrice'].sort_values(ascending=False) #calculates the correlation of new features with Sale Price and sorts them in descending order
print(correlations) #prints the correlation of new features with Sale Price

model_x = df[new_features] #creates a new dataframe model_x with the new features
model_y = df['SalePrice'] #creates a new dataframe model_y with Sale Price

X_train, X_test, y_train, y_test = train_test_split( #splits the data into training and testing sets
    model_x,
    model_y,
    test_size=0.2,
    random_state=42
) #splits the data into training and testing sets, 80% training and 20% testing

test_medians = X_test.median(numeric_only=True)
train_medians = X_train.median(numeric_only=True)
X_train = X_train.fillna(train_medians)
X_test = X_test.fillna(test_medians)

model = LinearRegression() #creates a linear regression model
model.fit(X_train, y_train) #fits the model to the training data

joblib.dump(model, "house_price_model.pkl")

y_pred = model.predict(X_test) #predicts Sale Price for the testing data
rmse = mean_squared_error(y_test, y_pred) ** 0.5 #calculates root mean squared error, 
#rmse is a measure of how well the model predicts Sale Price
r2 = r2_score(y_test, y_pred) #calculates r^2, indicates how well the model explains the variance in Sale Price
print("Root Mean Squared Error (RMSE):", rmse)
print("R-squared Score:", r2)

''' unique_count = df['RemodAge'].nunique()
print(f"Number of unique values in 'RemodAge': {unique_count}")

for feature in new_features: #iterates through each feature in new_features
    sns.scatterplot(data=df, x=feature, y='SalePrice')  #creates a scatter plot of each feature against Sale Price
    plt.title(f"{feature} vs Sale Price")  #sets the title of the plot
    plt.show() #displays the plot '''




