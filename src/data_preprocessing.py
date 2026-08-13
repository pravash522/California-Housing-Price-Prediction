import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# BASE_DIR = Path(__file__).resolve().parent.parent
# DATA_PATH = BASE_DIR / "data" / "california_housing.csv"

def load_data():
    df = pd.read_csv("../data/california_housing.csv")
    return df


def preprocess_data(df):
    X = df.drop("MedHouseVal", axis=1)
    y = df["MedHouseVal"]

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return (
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        scaler
    )