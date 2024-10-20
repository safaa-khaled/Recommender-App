import streamlit as st
import time

import backend as backend


# Basic webpage setup
st.set_page_config(
   page_title="Course Recommender System",
   layout="wide",
   initial_sidebar_state="expanded",
)

def train(model_name, params):
    # Start training course similarity model
    with st.spinner('Training...'):
        time.sleep(0.5)
        backend.train(model_name, params)
    st.success('Done!')

def predict(model_name, params):
    res = None
    # Start making predictions based on model name, test user ids, and parameters
    with st.spinner('Generating course recommendations: '):
        time.sleep(0.5)
        res = backend.predict(model_name, params)
    st.success('Recommendations generated!')
    return res

# ------ UI ------
# Sidebar
st.sidebar.title('Personalized Learning Recommender')
# Initialize the app
#selected_courses_df = init__recommender_app()

# Model selection selectbox
st.sidebar.subheader('1. Select recommendation models')
model_selection = st.sidebar.selectbox(
    "Select model:",
    backend.models
)

# Hyper-parameters for each model
params = {}
st.sidebar.subheader('2. Tune Hyper-parameters: ')
# Course similarity model
if model_selection == backend.models[0]:
    # Add a slide bar for selecting top courses
    top_courses = st.sidebar.slider('Top courses',
                                    min_value=0, max_value=100,
                                    value=10, step=1)
    # Add a slide bar for choosing similarity threshold
    course_sim_threshold = st.sidebar.slider('Course Similarity Threshold %',
                                             min_value=0, max_value=100,
                                             value=50, step=10)
    params['top_courses'] = top_courses
    params['sim_threshold'] = course_sim_threshold
# TODO: Add hyper-parameters for other models
# User profile model
elif model_selection == backend.models[1]:
    profile_sim_threshold = st.sidebar.slider('User Profile Similarity Threshold %',
                                              min_value=0, max_value=100,
                                              value=50, step=10)
    params['sim_threshold'] = profile_sim_threshold
# Clustering model
elif model_selection == backend.models[2]:
    cluster_no = st.sidebar.slider('Number of Clusters',
                                   min_value=0, max_value=50,
                                   value=20, step=1)
    params['cluster_no'] = cluster_no
else:
    pass


# Training
st.sidebar.subheader('3. Training: ')
training_button = st.sidebar.button("Train Model")
training_text = st.sidebar.text('')
# Start training process
if training_button:
    train(model_selection, params)

# Prediction
st.sidebar.subheader('4. Prediction')
# Start prediction process
pred_button = st.sidebar.button("Recommend New Courses")
if pred_button:
    # Create a new id for current user session
    res_df = predict(model_selection, params)
    st.table(res_df)
