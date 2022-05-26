import streamlit as st
import pickle
import numpy as np
import json
import gspread
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import credentials
import authorized_user

def run_streamlit():
    with st.sidebar:
        st.markdown(' ## Reference Links')
        st.write('[Spotipy](https://spotipy.readthedocs.io/)')
        st.write('[Spotify API](https://developer.spotify.com/documentation/web-api/)')
        st.write('[Spotify Web](https://open.spotify.com/)')
        st.markdown('## Contact Me')
        st.write('[Linkedin](https://www.linkedin.com/in/saumya-mundra/)')
        st.write('[Github](https://github.com/Saumya-svm)')
    
    st.header('Bike Rental')
    st.write("""
    Many U.S. cities have communal bike sharing stations where you can rent bicycles by the hour or day. Washington, D.C. is one of these cities. The District collects detailed data on the number of bicycles people rent by the hour and day. With this app, we will try to determine how many bikes the District would need to supply, given some details about time and weather, and some more details about the particular situation.

    """)
    
    st.markdown("""
    Here we have included the details of feature columns
    
- instant - A unique sequential ID number for each row
- dteday - The date of the rentals
- season - The season in which the rentals occurred
    - season (1:winter, 2:spring, 3:summer, 4:fall)
- yr - The year the rentals occurred
    - year (0: 2011, 1:2012)
- mnth - The month the rentals occurred
    - mnth (1 to 12)
- hr - The hour the rentals occurred
    - hour (0 to 23)
- holiday - Whether or not the day was a holiday
    - 1 is a holiday, 0 is not
- weekday - The day of the week (as a number, 0 to 6 starting from Sunday)
- workingday - Whether or not the day was a working day
    -  if day is neither weekend nor holiday is 1, otherwise is 0
- weathersit - The weather (as a categorical variable)
    - 1: Clear, Few clouds, Partly cloudy, Partly cloudy
    - 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
    - 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
    - 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
- temp - The temperature, on a 0-1 scale
- atemp - The adjusted temperature, on a 0-1 scale
- hum - The normalized humidity, on a 0-1 scale
- windspeed - The wind speed, on a 0-1 scale. The values were divided by 67
- time_label: time slot on a particular day.
    - {1: "between 06 to 12 hrs",
    
    2: "between 12 to 18 hrs",
    
    3: "between 18 to 00 hrs",
    
    4: "between 00 to 06 hrs"}
- casual - The number of casual riders (people who hadn't previously signed up with the bike sharing program)
- registered - The number of registered riders (people who had already signed up)
- cnt - The total number of bike rentals (casual + registered)

Kindly input values according to the range of values mentioned for the features above.
""")
    
    features = ['season','yr','mnth','hr','holiday','weekday','temp','atemp','hum','windspeed','time_label','weathersit','workingday']
    
    user_input = []
    try:
        for i in features:
            user_input.append(st.text_input(i))
             
        weather_list = [0,0,0,0]
        workingday_list = [0,0]
        
        if len(user_input) == len(features):
            for i in range(1,5):
                if i == int(user_input[-2]):
                    weather_list[i-1] = 1
                    break
            
            for i in range(2):
                if int(user_input[-1]) == i:
                    workingday_list[i] = 1
                    break
            
            user_input = user_input[:-2]+weather_list+workingday_list
            features =['season','yr','mnth','hr','holiday','weekday','temp','atemp','hum','windspeed','time_label','weathersit_1','weathersit_2','weathersit_3','weathersit_4','workingday_0','workingday_1']
            user_input = pd.DataFrame([user_input],columns=features)
            st.write(user_input)
            
        file = open('model.pkl','rb')
        model = pickle.load(file)
        st.write('The number of bikes needed by the district for this particular day is ',round(model.predict(user_input)[0]))
        
    except:
        st.write('Kindly input all the values above')


    
    with st.form(key='message_form'):
        st.markdown('## Send a Message')
        l = []
        form_submission = pd.DataFrame(columns=['Name','Email','Message'])
        name_input = st.text_input(label='Name',placeholder='Name')
        email_input = st.text_input(label='Email',placeholder='email')
        message = st.text_area(label='Message',placeholder='Message')
        submit_button = st.form_submit_button(label='Submit')
        dict = {}
        if submit_button:
            f = open('credentials.json')
            credentials = json.load(f)
            gc, authorized_user = gspread.oauth_from_dict(credentials,authorized_user.authorized_user)
            #authorized_user = json.loads(authorized_user)
            sh = gc.open_by_key(credentials.sheet_id)
            sheet = sh.get_worksheet(0)
            dict['Name'] = name_input
            dict['Email'] = email_input
            dict['Message'] = message
            l = list(dict.values())
            sheet.append_row(l)
   

run_streamlit()