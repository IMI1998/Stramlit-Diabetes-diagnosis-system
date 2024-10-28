import streamlit as st
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib
 
import seaborn as sns
import plotly.express as px

@st.cache
def load_data(data):
    df = pd.read_csv(data)
    return df

def run_eda_app():
    st.subheader("From Exploring Data Analysis")
    df = load_data("data\diabetes_data_upload.csv")
    freq_df = load_data("data\dfreqdist_of_age_data.csv")
    df_encoded = load_data("data\diabetes_data_upload_clean.csv")
   

    submenu = st.sidebar.selectbox("Submenu" , ["Descriptive" , "Plots"])
    if submenu == "Descriptive":
        st.dataframe(df)

        with st.expander("Data Types"):
            st.dataframe(df.dtypes)

        with st.expander("Descriptive Summery "):
            st.dataframe(df.describe())

        with st.expander("Class Distribution"):
            st.dataframe(df['class'].value_counts())

        with st.expander("Gender Distribution"):
            st.dataframe(df['Gender'].value_counts())

    elif submenu == "Plots":
        st.subheader("Plots")

        col1 , col2 = st.columns([1.900,1.10])   

        with col1:
            with st.expander("Dist Plot of Gender"):
                fig  = plt.figure()
                sns.countplot(df['Gender'])
                st.pyplot(fig) 

                gen_df = df['Gender'].value_counts().to_frame()
                gen_df = gen_df.reset_index()
                gen_df.columns = ["Gender Type" , "Counts"] 
                st.dataframe(gen_df)

                pl = px.pie(gen_df , names='Gender Type' , values='Counts' )
                st.plotly_chart(pl , use_container_width=True)

            with st.expander("Dist Plot of Class"):
                fig = plt.figure()
                sns.countplot(df['class'])
                st.pyplot(fig)        

        with col2:
            with st.expander("Gender Distribution"):
                st.dataframe(gen_df) 

            with st.expander("Class Distribution"):
                st.dataframe(df['class'].value_counts())            

        with st.expander("Frequency Dist of Age"):
            st.dataframe(freq_df) 
            p2 = px.bar(freq_df , x="Age" , y="count")
            st.plotly_chart(p2)

        with st.expander("Outlier Detection Plot"):
            fig = plt.figure()
            sns.boxplot(df['Age'])
            st.pyplot(fig)

            p3 = px.box(df , x="Age" , color="Gender")
            st.plotly_chart(p3 )

        with st.expander("Correlation Plot"):
            corr_matrix = df_encoded.corr()
            fig = plt.figure(figsize=(20,10))
            sns.heatmap(corr_matrix , annot=True)
            st.pyplot(fig)