#
import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
from PIL import Image
import requests
import pandas as pd
#
# st.set_page_config(layout="wide")
#
with st.container():
    selected = option_menu("Final Project", ["Predicting Breast Cancer in a patient","E-commerce Customer Segmentation", "Predicting Term Deposit Subscription by a client"], 
                           icons=['hospital' ,'cart2', 'bank'], menu_icon="brightness-alt-high", default_index=0)
    st.write("---")

    def load_lottieurl1(url): # Importing/Loading lottie URL
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    #up_left_column, up_right_column = st.columns(2)


    def BPrediction():
        st.header("""Predicting Breast Cancer in a patient""")
        title1 = """Abstract:
                    Breast cancer represents one of the diseases that make a high number of deaths every
                    year. It is the most common type of all cancers and the main cause of women's deaths
                    worldwide. Classification and data mining methods are an effective way to classify data.
                    Especially in the medical field, where those methods are widely used in diagnosis and
                    analysis to make decisions."""
        title2 = """Problem Statement:
                    Given the details of cell nuclei taken from breast mass, predict whether or not a patient
                    has breast cancer using the Ensembling Techniques. Perform necessary exploratory
                    data analysis before building the model and evaluate the model based on performance
                    metrics other than model accuracy."""
        title3 = """Dataset Information:
                    The dataset consists of several predictor variables and one target variable, Diagnosis.
                    The target variable has values 'Benign' and 'Malignant', where 'Benign' means that the
                    cells are not harmful or there is no cancer and 'Malignant' means that the patient has
                    cancer and the cells have a harmful effect."""
        st.write()
        st.write(title1)
        st.write(title2)
        st.write(title3)
        st.markdown("[Code : Predicting Breast Cancer](https://github.com/MaathanMethil/Predicting-Breast-Cancer-in-a-Patient/blob/main/Predicting%20Breast%20Cancer%20in%20a%20patient_GUVI01.ipynb)")
        st.write("---")
        lottie_coding3 = load_lottieurl1("https://lottie.host/acb222ab-625b-4ea6-83ed-d0e85f78c638/7eYE054R13.json")
        st_lottie(lottie_coding3, height=200, key="cod03")
        st.write("---") # 
        
        with st.container():
            up_left_column1, up_right_column1 = st.columns(2)
            #
            with up_left_column1:
                st.write("There are 357-'Benign' & 212-'Malignant'patients:")
                st.write("---")
                image_file = Image.open(r"C:\Users\DONMETHIL\Downloads\Xudemy\bp00.PNG")
                st.image(image_file, use_column_width=True)
                
                #     
            with up_right_column1:
                st.write("* Accuracy:")
                st.write("---")
                st.text("Accuracy on Training data : 96.04 %")
                st.text("Accuracy on Test data : 98.25 %")
                st.text("Accuracy score of the  SVC = 98.24 %")
                st.text("* Metrics Score on Training Data")
                st.text("Precision = 95.78 %")
                st.text("Recall = 93.52 %")
                st.text("F1 Score = 94.64 %")
                st.text("* Metrics Score on Test Data")
                st.text("Precision = 97.61 %")
                st.text("Recall = 97.61 %")
                st.text("F1 Score = 97.61 %")
                #
        with st.container():
           st.write("---")
           st.write("* Checking for Correlation: ")
           image_filec = Image.open(r"C:\Users\DONMETHIL\Downloads\Xudemy\bpcorr.PNG")
           st.image(image_filec, use_column_width=True) 
           #
        with st.container():
            st.write("---")
            up1_left_column1, up1_right_column1 = st.columns(2)
            #
            with up1_left_column1:
                st.write("* Bagging_Classifier:")
                st.write("---")
                st.text("Train_accuracy : 95.82")
                st.text("Test_accuracy : 96.49")
                st.text("Best Parameters: {'C': 5, 'kernel': 'linear'}")
                st.text("ROC Score : 96.72 %") 
                st.text("Precision Score : 93.18 %")
                st.text("Recall Score : 97.61 %")
                st.text("F1 Score : 95.34 %")
            #
            with up1_right_column1:
                st.write("* AdaBoost_Classifier:")
                st.write("---")
                st.text("Train_accuracy : 96.48")
                st.text("Test_accuracy : 99.20")  
                st.text("Highest Accuracy: 95.25 %")
                st.text("ROC Score : 97.72 %") 
                st.text("Precision Score : 96.18 %")
                st.text("Recall Score : 979.61 %")
                st.text("F1 Score : 98.34 %")
        #
        st.write("DATA SET")
        pbc = pd.read_csv(r"C:\Users\DONMETHIL\Downloads\Xudemy\Z-ML\cancerdata.csv")
        st.table(pbc.head(15))

    ###
    def ESegment():
        st.header("""E-commerce Customer Segmentation""")
        title5 = """Abstract:
                    A key challenge for e-commerce businesses is to analyze the trend in the market to increase their sales. 
                    The trend can be easily observed if the companies can group the customers; based on their activity on the e-commerce site. 
                    This grouping can be done by applying different criteria like previous orders, mostly searched brands and so on."""
        title6 = """Problem Statement:
                    Given the e-commerce data, use k-means clustering algorithm to cluster customers with similar interest."""
        title7 = """Dataset Information:
                    The data was collected from a well known e-commerce website over a period of time based on the customer’s search profile."""
        st.write()
        st.write(title5)
        st.write(title6)
        st.write(title7)
        st.markdown("[E-commerce Segmentation](https://github.com/MaathanMethil/E-Commerce-Customer-Segmentation/blob/main/E-commerce%20Customer%20Segmentation_GUVI_02.ipynb)")
        st.write("---")
        lottie_coding2 = load_lottieurl1("https://lottie.host/ed7a3629-c21e-4bcf-bafe-a0e325dc2700/wrF4JQsR0B.json")
        st_lottie(lottie_coding2, height=200, key="cod02")
        st.write("---")

        with st.container():
            up_left_column2, up_right_column2 = st.columns([1,2])
            #
            with up_left_column2:
                st.write("* Femail - 22054 & Mail - 5222")
                st.write("---")
                image_file0 = Image.open(r"C:\Users\DONMETHIL\Downloads\Xudemy\eccs.PNG")
                st.image(image_file0, width='auto', use_column_width=True)
                st.text("")
                #     
            with up_right_column2:
                st.write("* Overall Orders VS Gender wise Orders:")
                st.write("---")
                image_file1 = Image.open(r"C:\Users\DONMETHIL\Downloads\Xudemy\eccs1.PNG")
                st.image(image_file1, width='auto', use_column_width=True)
                st.text(" ")

        with st.container():
            st.text("* Top 10 Cust_ID based on Total Searches:")
            image_file2 = Image.open(r"C:\Users\DONMETHIL\Downloads\Xudemy\eccs2.PNG")
            st.image(image_file2, width='auto', use_column_width=True)
            st.text("* Total_Search Vs Cust_ID:")
            image_file2a = Image.open(r"C:\Users\DONMETHIL\Downloads\Xudemy\eccs3.PNG")
            st.image(image_file2a, width='auto', use_column_width=True)
            st.text("* Elbow Graph and Elbow Visualizer:")
            image_file2a = Image.open(r"C:\Users\DONMETHIL\Downloads\Xudemy\eccs4.PNG")
            st.image(image_file2a, width='auto', use_column_width=True)
            st.text(" ")
            #
        
        with st.container():
            st.header("* Cluster - 0 - Analysis:")
            image_file3 = Image.open(r"C:\Users\DONMETHIL\Downloads\Xudemy\eccs5.PNG")
            st.image(image_file3, width='auto', use_column_width=True)
            st.text(" ")
        
        with st.container():
            st.header("* Cluster - 1 - Analysis:")
            image_file4 = Image.open(r"C:\Users\DONMETHIL\Downloads\Xudemy\eccs6.PNG")
            st.image(image_file4, width='auto', use_column_width=True)
            st.text(" ")
        
        with st.container():
            st.header("* Cluster - 2 - Analysis:")
            image_file5 = Image.open(r"C:\Users\DONMETHIL\Downloads\Xudemy\eccs7.PNG")
            st.image(image_file5, width='auto', use_column_width=True)
            st.text(" ")
        
        st.write("DATA SET")
        daf = pd.read_excel(r"C:\Users\DONMETHIL\Downloads\Xudemy\Z-ML\custdata.xlsx")
        st.table(daf.head(10))

        with st.container():
            text001 = """Among 30000 customer, Cluster 0 has 12432 customers (Very Low past orders but done most searches) Cluster 1 has 9128 customers (Very High past orders and average searches) Cluster 2 has 8440 customers (Average past orders and average searches) * Cluster 0 has many customers but their past orders is only 7560. 
                         Cluster 1 is at the top based on past orders with 79885 orders which is more than 10 times of Cluster 0. 
                         Cluster 2 has least number of customers but has 37649 past orders which is almost 500% greater than cluster 0. 
                         Cluster 0 has done most number of searches with 81477 searches. Cluster 1 has 64573searches. Cluster 2 has least number of searches with 60093 searches."""
            st.write(text001)
        
    ###
    def DPrediction():
        st.header("""Predicting Term Deposit Subscription by a client""")
        title9 = """Abstract:
                    Marketing campaigns are characterized by focusing on the customer needs and their
                    overall satisfaction. Nevertheless, there are different variables that determine whether a
                    marketing campaign will be successful or not. There are certain variables that we need
                    to take into consideration when making a marketing campaign.
                    A Term deposit is a deposit that a bank or a financial institution offers with a fixed rate
                    (often better than just opening a deposit account) in which your money will be returned
                    back at a specific maturity time."""
        title10 = """Problem Statement:
                    Predict if a customer subscribes to a term deposits or not, when contacted by a
                    marketing agent, by understanding the different features and performing predictive
                    analytics"""
        title11 = """Dataset Information:
                    The data is related with direct marketing campaigns of a Portuguese banking
                    institution. The marketing campaigns were based on phone calls. Often, more than one
                    contact to the same client was required, in order to assess if the product (bank term
                    deposit) would be ('yes') or not ('no') subscribed.
                    The dataset consists of several predictor variables and one target variable, Outcome.
                    Predictor variables includes the age, job, marital status, and so on"""
        st.write()
        st.write(title9)
        st.write(title10)
        st.write(title11)
        st.markdown("[Predicting Term Deposit](https://github.com/MaathanMethil/Predicting-Term-Deposit-Subscription-by-a-client/blob/main/Predicting%20Term%20Deposit%20Subscription%20by%20a%20client_GUVI03.ipynb)")
        st.write("---")
        lottie_coding1 = load_lottieurl1("https://lottie.host/fea04366-64b6-40e2-9cd8-d4f22f70194c/fYKGyM41v6.json")
        st_lottie(lottie_coding1, height=200, key="cod01")
        st.write("---")

        with st.container():
            up_left_column3, up_right_column3 = st.columns(2)
            #
            with up_left_column3:
                st.write("* Age wise visualization:")
                st.write("---")
                image_file5 = Image.open(r"C:\Users\DONMETHIL\Downloads\Xudemy\pds.PNG")
                st.image(image_file5, width='auto', use_column_width=True)
                st.text(" ")
                #     
            with up_right_column3:
                st.write("* People who has taken Subscription:")
                st.write("---")
                image_file5a = Image.open(r"C:\Users\DONMETHIL\Downloads\Xudemy\pds01.PNG")
                st.image(image_file5a, width='auto', use_column_width=True)
                st.text(" ")
        st.write("---")
        with st.container():
            st.write("* Subscribed Vs Marital Status Vs Job")
            image_file6 = Image.open(r"C:\Users\DONMETHIL\Downloads\Xudemy\pds02.PNG")
            st.image(image_file6, width='auto', use_column_width=True)
            st.write("---")
            #
        with st.container():
            st.write("* Subscribed Vs Job")
            image_file6 = Image.open(r"C:\Users\DONMETHIL\Downloads\Xudemy\pds03.PNG")
            st.image(image_file6, width='auto', use_column_width=True)
            st.write("---")
            #
        with st.container():
            up3_left_column3, up3_right_column3 = st.columns(2)
            #
            with up3_left_column3:
                st.write("* Subscription in accordance with Education:")
                st.write("---")
                image_file5 = Image.open(r"C:\Users\DONMETHIL\Downloads\Xudemy\pds04.PNG")
                st.image(image_file5, width='auto', use_column_width=True)
                st.text(" ")
                #   
            with up3_right_column3:
                st.write("* Subscription in accordance with Marital Status:")
                st.write("---")
                image_file5 = Image.open(r"C:\Users\DONMETHIL\Downloads\Xudemy\pds05.PNG")
                st.image(image_file5, width='auto', use_column_width=True)
                st.text(" ")
                #   
        with st.container():
            st.text("* SVM using Polynomial kernal with degree of polynomial = 2")
            st.text("Accuracy Of SVM sigmoid kernal:  89.67 %")
            st.text("Precision_Score : 0.98 ")
            st.text("Recall_Score : 0.91 ")
            st.text("F1_Score : 0.94 ")
            #
        st.write("DATA SET")
        dff = pd.read_csv(r"C:\Users\DONMETHIL\Downloads\Xudemy\Z-ML\bank-additional-full.csv")
        st.table(dff.head(10))
         
    ###

    if selected == "Predicting Breast Cancer in a patient":
        BPrediction()
        st.write("---")
    elif selected == "E-commerce Customer Segmentation":
        ESegment()
        st.write("---")
    elif selected == "Predicting Term Deposit Subscription by a client":
        DPrediction()
        st.write("---")
    #

#
#

#


