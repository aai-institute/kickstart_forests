import config 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


if __name__ == '__main__':

    df = pd.read_csv(config.csv_data_path(), index_col= 0)

    df.dropna(inplace = True)

    df.drop('treeSpecies', axis = 1, inplace= True)

    y = df['lai']

    X = df.iloc[:,1:12]

    X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42)

    m = LinearRegression()


    m.fit(X_train, y_train)

    m.score(X_train, y_train)

    m.score(X_test, y_test)


    rf = RandomForestRegressor()

    rf.fit(X_train, y_train)

    rf.score(X_train, y_train)
    
    print(rf.score(X_test, y_test))

