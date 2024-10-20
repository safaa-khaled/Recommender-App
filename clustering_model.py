# import required library
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# prepare dataset
def prepare_dataset():
    user_profile_url = "data/user_profile.csv"
    user_profile_df = pd.read_csv(user_profile_url)
    feature_names = list(user_profile_df.columns[1:])
    # Standardizing the selected features (feature_names) in the user_profile_df DataFrame
    scaler = StandardScaler()
    user_profile_df[feature_names] = scaler.fit_transform(user_profile_df[feature_names])

    # splits features and user_ids
    features = user_profile_df.loc[:, user_profile_df.columns != 'user']
    user_ids = user_profile_df.loc[:, user_profile_df.columns == 'user']
    return features,user_ids

features, user_ids = prepare_dataset()

# train function
def train_kmeans(dataframe, params):
    k = params['cluster_no']
    model = KMeans(n_clusters=k, random_state=23)
    model.fit(dataframe)
    return model

# predict function
def predict_kmeans(cluster_df):

    test_user_url = "data/ratings.csv"
    test_users_df = pd.read_csv(test_user_url)[['user', 'item']]
    test_users_df = test_users_df.iloc[0:5,:]
    test_users_labelled = pd.merge(test_users_df, cluster_df, left_on='user', right_on='user')
    courses_cluster = test_users_labelled[['item', 'cluster']]
    # Adding a new column 'count' with a value of 1 for each row in the courses_cluster DataFrame
    courses_cluster['count'] = [1] * len(courses_cluster)
    courses_cluster_grouped = courses_cluster.groupby(['cluster', 'item']).agg(
        enrollments=('count', 'sum')).reset_index()
    threshold = 10
    test_user_ids = list(test_users_labelled['user'])
    users_list = list()
    courses_list = list()
    enrollment_list = list()

    # for each user
    for user_id in test_user_ids:
        # get user cluster
        cluster = cluster_df.loc[cluster_df['user'] == user_id]['cluster']
        cluster = cluster.iloc[0]
        # get all courses in cluster with enrollment threshold
        cluster_subset = courses_cluster_grouped[
            (courses_cluster_grouped['cluster'] == cluster) & (courses_cluster_grouped['enrollments'] > threshold)]
        cluster_courses = set(cluster_subset['item'])
        # get enrolled courses for this user
        user_subset = test_users_labelled[test_users_labelled['user'] == user_id]
        enrolled_courses = set(user_subset['item'])
        # get unseen courses for this user
        unseen_courses = cluster_courses.difference(enrolled_courses)
        for course in unseen_courses:
            users_list.append(user_id)
            courses_list.append(course)
            enrollment_list.append(threshold)
    return users_list,courses_list,enrollment_list

# combine users with their clusters
def combine_cluster_labels(user_ids, labels):
    # Convert labels to a DataFrame
    labels_df = pd.DataFrame(labels)
    # Merge user_ids DataFrame with labels DataFrame based on index
    cluster_df = pd.merge(user_ids, labels_df, left_index=True, right_index=True)
    # Rename columns to 'user' and 'cluster'
    cluster_df.columns = ['user', 'cluster']
    return cluster_df



