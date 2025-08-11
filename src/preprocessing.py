import pandas as pd

def ageupdate(Pclass, Age):
    if pd.isnull(Age):
        if Pclass == 1:
            return 40
        elif Pclass == 2:
            return 28
        elif Pclass == 3:
            return 24
        else:
            return 0
    else:
        return Age

def preprocess_data(csv_path):
    df = pd.read_csv(csv_path)
    df.drop(['Name', 'PassengerId', 'Ticket', 'Fare', 'Cabin', 'Embarked'], axis=1, inplace=True)
    y = df['Survived']
    X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']]
    X.loc[:, 'Age'] = X.apply(lambda row: ageupdate(row['Pclass'], row['Age']), axis=1)

    gender = pd.get_dummies(X['Sex'], drop_first=False)
    X = pd.concat([X.drop('Sex', axis=1), gender], axis=1)

    room_class = pd.get_dummies(X['Pclass'], prefix='class', drop_first=False)
    X = pd.concat([X.drop('Pclass', axis=1), room_class], axis=1)

    siblings = pd.get_dummies(X['SibSp'], prefix='s', drop_first=False)
    X = pd.concat([X.drop('SibSp', axis=1), siblings], axis=1)

    parentchild = pd.get_dummies(X['Parch'], prefix='p', drop_first=False)
    X = pd.concat([X.drop('Parch', axis=1), parentchild], axis=1)

    return X, y
