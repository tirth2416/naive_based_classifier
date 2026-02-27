import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

st.title("ML Model Evaluation Panel")

# Upload dataset
uploaded_file = st.file_uploader("Upload Dataset (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.dataframe(df.head())

    # -----------------------------
    # 🔍 EXPLORATORY DATA ANALYSIS
    # -----------------------------
    st.subheader("Exploratory Data Analysis")

    st.write("Dataset Shape:", df.shape)

    st.write("Data Types:")
    st.write(df.dtypes)

    st.write("Missing Values:")
    st.write(df.isnull().sum())

    st.write("Statistical Summary:")
    st.write(df.describe())

    # Correlation Matrix
    numeric_df = df.select_dtypes(include=["int64", "float64"])

    if not numeric_df.empty:
        st.write("Correlation Matrix:")
        fig, ax = plt.subplots()
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    # Distribution Plot
    st.subheader("Feature Distribution")
    column_to_plot = st.selectbox("Select column to visualize", df.columns)

    fig2, ax2 = plt.subplots()
    sns.histplot(df[column_to_plot], kde=True, ax=ax2)
    st.pyplot(fig2)

    # -----------------------------
    # ✅ Problem Type Selection
    # -----------------------------
    problem_type = st.radio(
        "Select Problem Type",
        ["Classification", "Regression"]
    )

    # ------------------------------------
    # CLASSIFICATION SECTION
    # ------------------------------------
    if problem_type == "Classification":

        st.subheader("Classification Settings")

        possible_targets = [
            col for col in df.columns
            if df[col].dtype == "object" or df[col].nunique() < 15
        ]

        target = st.selectbox("Select Target Variable", possible_targets)

        features = st.multiselect(
            "Select Feature Columns",
            [col for col in df.columns if col != target]
        )

        col1, col2 = st.columns(2)
        train_size = col1.number_input("Train %", 10, 90, 70)
        test_size = col2.number_input("Test %", 10, 90, 30)

        if st.button("Evaluate Classification Model"):

            from sklearn.naive_bayes import GaussianNB
            from sklearn.metrics import accuracy_score, confusion_matrix

            X = df[features]
            y = df[target]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size/100, random_state=42
            )

            model = GaussianNB()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)

            st.success(f"Accuracy: {acc:.2f}")
            st.write("Confusion Matrix:")
            st.write(cm)

    # ------------------------------------
    # REGRESSION SECTION
    # ------------------------------------
    else:

        st.subheader("Regression Settings")

        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

        target = st.selectbox("Select Target Variable", numeric_cols)

        features = st.multiselect(
            "Select Feature Columns",
            [col for col in numeric_cols if col != target]
        )

        col1, col2 = st.columns(2)
        train_size = col1.number_input("Train %", 10, 90, 70)
        test_size = col2.number_input("Test %", 10, 90, 30)

        if st.button("Evaluate Regression Model"):

            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import mean_squared_error, r2_score

            X = df[features]
            y = df[target]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size/100, random_state=42
            )

            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)

            st.success(f"R2 Score: {r2:.2f}")
            st.write(f"Mean Squared Error: {mse:.2f}")
