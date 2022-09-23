import streamlit as st
import pickle
import numpy as np

# import the model
pipe = pickle.load(open('model.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))


def main():
    string = "Price Predictor"
    st.set_page_config(page_title=string, page_icon="📱")
    st.title("Mobile Price Predictor")
    st.image(
        "https://dictionary.cambridge.org/images/thumb/mobile_noun_002_23642.jpg?version=5.0.247",
        width=170 # Manually Adjust the width of the image as per requirement
    )
    st.write('')
    st.write('')

    brand = st.selectbox('Select Mobile Company', df['brand'].unique())
    model = st.selectbox('Select Model', df['model_name'].unique())
    ops = st.selectbox('Operating System', df['ops'].unique())
    display = st.selectbox('Display Type', ['AMOLED', 'LCD', 'HD+','LED','Super Amoled Plus','OLED','FHD +'])
    colour = st.selectbox('Select colour', df['colour'].unique())
    rom = st.selectbox('Storage (ROM) in GB', df['storage'].unique())
    ram = st.selectbox('RAM in GB', df['ram'].unique())
    battery = st.selectbox('Battery in MAh', df['battery'].unique())
    weight = st.selectbox('Weight in grams', df['weight'].unique())
    display_size = st.selectbox('Display Size in cm', df['display_size'].unique())
    Review = st.selectbox('Select Rating given by customers', [1,2,2.5,3,3.5,3.8,4,4.2,4.3,4.5,4.6,4.8,5])

    if st.button('Predict Price'):
        query = np.array(
            [brand, model, ops, display, colour, rom, ram, battery, weight, display_size, Review])

        query = query.reshape(1, 11)
        st.title("The predicted price  " + str(int(np.exp(pipe.predict(query)[0]))) + '  Rupees')


if __name__ == '__main__':
    main()
