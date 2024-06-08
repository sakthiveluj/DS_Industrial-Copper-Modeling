import pandas as pd
import streamlit as st
import numpy as np
import pickle as pk
import math
from streamlit_option_menu import option_menu

st.set_page_config(layout="wide")
st.write("""
<div style='text-align:center'>
    <h1 style='color:#FB04F7;'>Industrial Copper Modeling Application</h1>
</div>
""", unsafe_allow_html=True)

#streamlit page
with st.sidebar:
    select_option=option_menu("Menu",["About","Selling Price Predictor","Status Predictor"])
if select_option=="About":


    # CSS for background image
    page_bg_img = '''
    <style>
    body {
        background-image: url("https://example.com/your-background-image.jpg");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    .main-content {
        background-color: rgba(255, 255, 255, 0.8); /* White background with some transparency */
        padding: 20px;
        border-radius: 10px;
        max-width: 800px;
        margin: auto;
    }
    </style>
    '''

    # Inject CSS with HTML
    st.markdown(page_bg_img, unsafe_allow_html=True)

    # About Page Content
    st.markdown('<div class="main-content">', unsafe_allow_html=True)
    st.title("About the Industrial Copper Model Application")
    st.write("""
    Welcome to the Industrial Copper Model Application!

    This application leverages advanced machine learning models to predict the selling price of copper and determine its status. Our 
    primary goal is to provide accurate and reliable predictions to assist industries in making informed decisions about copper trading 
    and management.

    ### Features
    - **Copper Price Prediction**: Utilizing a Random Forest Regressor, we predict the selling price of copper with an accuracy of 93%. 
    This high level of accuracy ensures that our users receive precise pricing information.
    - **Copper Status Classification**: Our Random Forest Classifier helps in determining the status of copper (e.g., Good or Bad), 
    aiding in quality control and inventory management.

    ### How It Works
    The application takes input features such as market conditions, copper quality metrics, and other relevant data points to make 
    predictions. These features are processed by our machine learning models, which have been trained on historical data to ensure 
    high accuracy and reliability.

    ### Our Models
    - **Random Forest Regressor**: This model is used for predicting the selling price of copper. It works by building multiple 
    decision trees and combining their results to improve accuracy and reduce overfitting.
    - **Random Forest Classifier**: This model is used for classifying the status of copper. It operates similarly to the regressor, 
    using an ensemble of decision trees to enhance prediction performance.

    ### Benefits
    - **Accurate Predictions**: Our models provide precise and reliable predictions, helping users make informed decisions.
    - **User-Friendly Interface**: The application is designed to be intuitive and easy to use, allowing users to input data and receive 
    predictions quickly.
    - **Advanced Machine Learning**: By leveraging state-of-the-art machine learning techniques, we ensure that our predictions are based 
    on the latest advancements in the field.

    ### Conclusion
    The Industrial Copper Model Application is a powerful tool for anyone involved in the copper industry. Whether you're a trader, 
    manufacturer, or quality control specialist, our application provides the insights you need to succeed. Explore the application and 
    discover how it can help you achieve your goals in the copper market.

    Thank you for using our application!
    """)
    st.markdown('</div>', unsafe_allow_html=True)

elif select_option=="Selling Price Predictor":
    st.write("""
    <div style='text-align:center'>
        <h3 style='color:#FB1004;'>Selling Price Predictor</h3>
    </div>
    """, unsafe_allow_html=True)
    status_options = ['Won', 'Draft', 'To be approved', 'Lost', 'Not lost for AM', 'Wonderful', 'Revised', 'Offered', 'Offerable']
    item_type_options = ['W', 'WI', 'S', 'Others', 'PL', 'IPL', 'SLAWR']
    country_options = [28., 25., 30., 32., 38., 78., 27., 77., 113., 79., 26., 39., 40., 84., 80., 107., 89.]
    application_options = [10., 41., 28., 59., 15., 4., 38., 56., 42., 26., 27., 19., 20., 66., 29., 22., 40., 25., 67., 79., 3., 99., 2., 5., 39., 69., 70., 65., 58., 68.]
    product=['611112', '611728', '628112', '628117', '628377', '640400', '640405', '640665', 
                    '611993', '929423819', '1282007633', '1332077137', '164141591', '164336407', 
                    '164337175', '1665572032', '1665572374', '1665584320', '1665584642', '1665584662', 
                    '1668701376', '1668701698', '1668701718', '1668701725', '1670798778', '1671863738', 
                    '1671876026', '1690738206', '1690738219', '1693867550', '1693867563', '1721130331', '1722207579']
    # Define the widgets for user input
    st.write( f'<h5 style="color:rgb(120, 251, 4);">NOTE: Min & Max given for reference, you can enter any value</h5>', unsafe_allow_html=True )
    col1,col2,col3=st.columns([5,2,5])
    with col1:
        status = st.selectbox("Status", status_options,key=1)
        item_type = st.selectbox("Item Type", item_type_options,key=2)
        country = st.selectbox("Country", sorted(country_options),key=3)
        application = st.selectbox("Application", sorted(application_options),key=4)
        product_ref = st.selectbox("Product Reference", product,key=5)
    with col3:
        quantity_tons = st.text_input("Enter Quantity Tons (Min:611728 & Max:1722207579)")
        thickness = st.text_input("Enter thickness (Min:0.18 & Max:400)")
        width = st.text_input("Enter width (Min:1, Max:2990)")
        customer = st.text_input("customer ID (Min:12458, Max:30408185)")
    submit_button = st.button(label="PREDICT SELLING PRICE")
    st.markdown("""
                        <style>
                        div.stButton > button:first-child {
                            background-color: #FBF704;
                            color: red;
                            width: 100%;
                        }
                        </style>
        """, unsafe_allow_html=True)
    def fun():
        new_sample= np.array([[quantity_tons,application,thickness,width,country,customer,product_ref,item_type,status]])
        new=pd.DataFrame(new_sample,columns=["quantity_tons_log","application","thickness_log","width","country","customer","product_ref","item type","status"])
        with open(r"RandomForestRegressor.pkl", 'rb') as file:
            loaded_model = pk.load(file)
        new_pred = loaded_model.predict(new)
        reversed_value = math.exp(new_pred)
        return(reversed_value)
    if submit_button:
        st.write(fun())
elif select_option=="Status Predictor":
    st.write("""
    <div style='text-align:center'>
        <h3 style='color:#FB1004;'>Status Predictor</h3>
    </div>
    """, unsafe_allow_html=True)
    item_type_options = ['W', 'WI', 'S', 'Others', 'PL', 'IPL', 'SLAWR']
    country_options = [28., 25., 30., 32., 38., 78., 27., 77., 113., 79., 26., 39., 40., 84., 80., 107., 89.]
    application_options = [10., 41., 28., 59., 15., 4., 38., 56., 42., 26., 27., 19., 20., 66., 29., 22., 40., 25., 67., 79., 3., 99., 2., 5., 39., 69., 70., 65., 58., 68.]
    product=['611112', '611728', '628112', '628117', '628377', '640400', '640405', '640665', 
                    '611993', '929423819', '1282007633', '1332077137', '164141591', '164336407', 
                    '164337175', '1665572032', '1665572374', '1665584320', '1665584642', '1665584662', 
                    '1668701376', '1668701698', '1668701718', '1668701725', '1670798778', '1671863738', 
                    '1671876026', '1690738206', '1690738219', '1693867550', '1693867563', '1721130331', '1722207579']
    st.write( f'<h5 style="color:rgb(120, 251, 4);">NOTE: Min & Max given for reference, you can enter any value</h5>', unsafe_allow_html=True )
    col1,col2,col3=st.columns([5,2,5])
    with col1:
                st.write(' ')
                item_type = st.selectbox("Item Type", item_type_options,key=2)
                country = st.selectbox("Country", sorted(country_options),key=3)
                application = st.selectbox("Application", sorted(application_options),key=4)
                product_ref = st.selectbox("Product Reference", product,key=5)
    with col3:               
                cselling = st.text_input("Selling Price (Min:1, Max:100001015)")
                quantity_tons = st.text_input("Enter Quantity Tons (Min:611728 & Max:1722207579)")
                thickness = st.text_input("Enter thickness (Min:0.18 & Max:400)")
                width = st.text_input("Enter width (Min:1, Max:2990)")
                customer = st.text_input("customer ID (Min:12458, Max:30408185)")
    submit_button = st.button(label="PREDICT STATUS")
    st.markdown("""
                    <style>
                    div.stButton > button:first-child {
                        background-color: #FBF704;
                        color: red;
                        width: 100%;
                    }
                    </style>
                """, unsafe_allow_html=True)
    def fun1():
        new_sample= np.array([[quantity_tons,application,thickness,width,country,customer,product_ref,item_type,cselling]])
        new=pd.DataFrame(new_sample,columns=["quantity_tons_log","application","thickness_log","width","country","customer","product_ref","item type","selling_price_log"])
        with open(r"RandomForestClassifier.pkl", 'rb') as file:
            loaded_model = pk.load(file)
        new_pred = loaded_model.predict(new)
        return(new_pred)
    if submit_button:
        a=fun1()
        d=pd.DataFrame(a,columns=["a"])
        df=d["a"]
        st.write(df[0])