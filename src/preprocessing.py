import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from config import TEST_SIZE, RANDOM_STATE

def preprocess(df):
    df = df.copy()

    # Convert date columns
    df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
    df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])
    df['WaitingDays'] = (df['AppointmentDay'] - df['ScheduledDay']).dt.days

    # Convert target to numeric
    df['No-show'] = df['No-show'].map({'No': 0, 'Yes': 1})

    # Features and target
    X = df.drop(columns=['No-show'])
    y = df['No-show']

    # Categorical and numeric columns
    categorical = ['Gender', 'Neighbourhood']
    numeric = [
        'Age', 'Scholarship', 'Hipertension', 'Diabetes',
        'Alcoholism', 'Handcap', 'SMS_received', 'WaitingDays'
    ]

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical),
            ('num', 'passthrough', numeric)
        ]
    )

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Fit-transform
    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
    X_train = pipeline.fit_transform(X_train)
    X_test = pipeline.transform(X_test)

    # Extract feature names
    feature_names = (
        pipeline.named_steps['preprocessor']
        .named_transformers_['cat']
        .get_feature_names_out(categorical)
        .tolist()
        + numeric
    )

    return X_train, X_test, y_train, y_test, feature_names

