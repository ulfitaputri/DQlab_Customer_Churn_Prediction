import streamlit as st
import pandas as pd
import pickle

with open("best_model_churn.pkl", "rb") as file:
    loaded_model = pickle.load(file)


def valreplace(data):
    replace_dict = {
        "Yes" or "yes": 1,
        "No" or "no": 0,
        "Female" or "female": 1,
        "Male" or "male": 0,
    }
    return data.replace(replace_dict)


def main():
    add_selectbox = st.sidebar.selectbox(
        "How would you like to predict?", ("Online", "Batch")
    )
    st.sidebar.info("**This app is created to predict Customer Churn at DQlab Telco**")
    st.title("DQLab Telco Customer Churn Prediction")
    st.header("Customer Data Input")
    if add_selectbox == "Online":
        gender = st.selectbox("**Customer's gender identity**", ("Female", "Male"))
        SeniorCitizen = st.selectbox("**Is the customer a senior citizen?**", ("Yes", "No"))
        Partner = st.selectbox("**Does the customer have a partner?**", ("Yes", "No"))
        tenure = st.number_input("**Tenure (in month)**", 0, 130, 0)
        PhoneService = st.selectbox(
            "**Does the customer have a phone service?**", ("Yes", "No")
        )
        StreamingTV = st.selectbox(
            "**Does the customer have a streaming TV?**", ("Yes", "No")
        )
        InternetService = st.selectbox(
            "**Does the customer have an interner service**", ("Yes", "No")
        )
        PaperlessBilling = st.selectbox(
            "**Does the customer have paperless billing?**", ("Yes", "No")
        )
        MonthlyCharges = st.number_input("**Customer's monthly charges**", 0, 170, 0)
        TotalCharges = MonthlyCharges * tenure
        preview_data = pd.DataFrame(
            [
                [
                    gender,
                    SeniorCitizen,
                    Partner,
                    tenure,
                    PhoneService,
                    StreamingTV,
                    InternetService,
                    PaperlessBilling,
                    MonthlyCharges,
                    TotalCharges,
                ]
            ],
            columns=[
                "gender",
                "SeniorCitizen",
                "Partner",
                "tenure",
                "PhoneService",
                "StreamingTV",
                "InternetService",
                "PaperlessBilling",
                "MonthlyCharges",
                "TotalCharges",
            ],
        )
        if st.button("Preview Data"):
            st.dataframe(preview_data)
        data_predict = valreplace(preview_data)
        if st.button("Predict"):
            predicted_churn = loaded_model.predict(data_predict)
            st.success("**Churn : Yes**" if predicted_churn == 1 else "**Churn : No**")
    if add_selectbox == "Batch":
        st.warning(
            "Before uploading the data, make sure it looks like this or you can download the template"
        )
        data_example = pd.read_csv("data_example.csv", sep=";")
        st.dataframe(data_example)

        @st.cache_data
        def convert_df(df):
            return df.to_csv(index=False, sep = ";").encode("utf-8")

        csv = convert_df(data_example)
        st.download_button(
            label="Download Template",
            data=csv,
            file_name="template.csv",
            mime="text/csv",
        )
        file_upload = st.file_uploader("Upload CSV file for predictions", type=["csv"])
        if file_upload is not None:
            new_data = pd.read_csv(file_upload, sep=(";"))
            new_data["TotalCharges"] = new_data["tenure"] * new_data["MonthlyCharges"]
            if st.button("Predict"):
                if new_data.isnull().values.any():
                    st.dataframe(new_data)
                    st.warning(
                        "Data contains null, please check your data and upload again"
                    )
                else:
                    data_predict = valreplace(new_data)
                    predictions = []
                    for index, row in data_predict.iterrows():
                        predicted_churn = loaded_model.predict(
                            row.values.reshape(1, -1)
                        )
                        predictions.append("Yes" if predicted_churn[0] == 1 else "No")
                    new_data["Predicted_Churn"] = predictions
                    st.dataframe(new_data)
                    churn_rate = round(
                        (
                            new_data["Predicted_Churn"].value_counts()["Yes"]
                            / len(new_data)
                        )
                        * 100,
                        2,
                    )
                    st.success(f"**Churn Rate :** {churn_rate}%")
                    csv = convert_df(new_data)
                    st.download_button(
                        label="Download Data",
                        data=csv,
                        file_name="customer_churn.csv",
                        mime="text/csv",
                    )


if __name__ == "__main__":
    main()
