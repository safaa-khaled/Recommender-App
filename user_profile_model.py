import pandas as pd
import numpy as np


# prepare dataset
def prepare_dataset():
    test_user_url = "data/ratings.csv"
    test_users_df = pd.read_csv(test_user_url)
    test_users = test_users_df.groupby(['user']).max().reset_index(drop=False)
    test_users = test_users.iloc[0:5,:]

    profile_genre_url = "data/user_profile.csv"
    profile_df = pd.read_csv(profile_genre_url)

    course_genre_url = "data/course_genre.csv"
    course_genres_df = pd.read_csv(course_genre_url)

    return test_users, profile_df, course_genres_df


test_users, profile_df, course_genres_df = prepare_dataset()

def generate_recommendation_scores(score_threshold):
    """
    Generate recommendation scores for users and courses.

    Returns:
    users (list): List of user IDs.
    courses (list): List of recommended course IDs.
    scores (list): List of recommendation scores.
    """

    users = []      # List to store user IDs
    courses = []    # List to store recommended course IDs
    scores = []     # List to store recommendation scores

    test_user_ids = test_users['user'].to_list()
    all_courses = set(course_genres_df['COURSE_ID'].values)
    # Iterate over each user ID in the test_user_ids list
    for user_id in test_user_ids:
        # Get the user profile data for the current user
        test_user_profile = profile_df[profile_df['user'] == user_id]

        # Get the user vector for the current user id (replace with your method to obtain the user vector)
        test_user_vector = test_user_profile.iloc[0, 1:].values

        # Get the known course ids for the current user
        enrolled_courses = test_users[test_users['user'] == user_id]['item'].to_list()

        # Calculate the unknown course ids
        unknown_courses = all_courses.difference(enrolled_courses)

        # Filter the course_genres_df to include only unknown courses
        unknown_course_df = course_genres_df[course_genres_df['COURSE_ID'].isin(unknown_courses)]
        unknown_course_ids = unknown_course_df['COURSE_ID'].values

        # Calculate the recommendation scores using dot product
        recommendation_scores = np.dot(unknown_course_df.iloc[:, 2:].values, test_user_vector)

        # Append the results into the users, courses, and scores list
        for i in range(0, len(unknown_course_ids)):
            score = recommendation_scores[i]

            # Only keep the courses with high recommendation score
            if score >= score_threshold:
                users.append(user_id)
                courses.append(unknown_course_ids[i])
                scores.append(recommendation_scores[i])

    return users, courses, scores


#generate_recommendation_scores(10)