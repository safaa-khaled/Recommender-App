import pandas as pd
import numpy as np


# prepare dataset
def prepare_dataset():
    sim_url = "data/sim.csv"
    sim_df = pd.read_csv(sim_url)
    # load the course content and BoW dataset
    course_url = "data/course_processed.csv"
    course_df = pd.read_csv(course_url)
    bow_url = "data/courses_bows.csv"
    bow_df = pd.read_csv(bow_url)
    test_user_url = "data/ratings.csv"
    test_users_df = pd.read_csv(test_user_url)
    test_users = test_users_df.groupby(['user']).max().reset_index(drop=False)
    test_users = test_users.iloc[0:5,:]
    #test_user_ids = test_users['user'].to_list()
    return sim_df, course_df, bow_df, test_users


sim_df, course_df, bow_df, test_users = prepare_dataset()


# Create course id to index and index to id mappings
def get_doc_dicts(bow_df):
    # Group the DataFrame by course index and ID, and get the maximum value for each group
    grouped_df = bow_df.groupby(['doc_index', 'doc_id']).max().reset_index(drop=False)
    # Create a dictionary mapping indices to course IDs
    idx_id_dict = grouped_df[['doc_id']].to_dict()['doc_id']
    # Create a dictionary mapping course IDs to indices
    id_idx_dict = {v: k for k, v in idx_id_dict.items()}
    # Clean up temporary DataFrame
    del grouped_df
    return idx_id_dict, id_idx_dict


def generate_recommendations_for_one_user(enrolled_course_ids, unselected_course_ids, id_idx_dict, sim_matrix,
                                          threshold):
    # Create a dictionary to store your recommendation results
    res = {}
    # Iterate over enrolled courses
    for enrolled_course in enrolled_course_ids:
        # Iterate over unselected courses
        for unselect_course in unselected_course_ids:
            # Check if both enrolled and unselected courses exist in the id_idx_dict
            if enrolled_course in id_idx_dict and unselect_course in id_idx_dict:
                # Initialize similarity value
                sim = 0
                # Find the two indices for each enrolled_course and unselect_course, based on their two ids
                # Calculate the similarity between an enrolled_course and an unselect_course
                # e.g., Course ML0151EN's index is 200 and Course ML0101ENv3's index is 158
                idx1 = id_idx_dict[enrolled_course]
                idx2 = id_idx_dict[unselect_course]
                # Find the similarity value from the sim_matrix
                sim = sim_matrix[idx1][idx2]
                # Check if the similarity exceeds the threshold
                if sim > threshold:
                    # Update recommendation dictionary with course ID and similarity score
                    if unselect_course not in res:
                        # If the unselected course is not already in the recommendation dictionary (`res`), add it.
                        res[unselect_course] = sim
                    else:
                        # If the unselected course is already in the recommendation dictionary (`res`), compare the similarity score.
                        # If the current similarity score is greater than or equal to the existing similarity score for the course,
                        # update the similarity score in the recommendation dictionary (`res`) with the current similarity score.
                        if sim >= res[unselect_course]:
                            res[unselect_course] = sim

    # Sort the results by similarity
    res = {k: v for k, v in sorted(res.items(), key=lambda item: item[1], reverse=True)}
    # Return the recommendation dictionary
    return res


def generate_recommendations_for_all(threshold):
    users = []
    courses = []
    sim_scores = []

    sim_matrix = sim_df.to_numpy()
    #get users ids
    test_user_ids = test_users['user'].to_list()
    # set id to index and index to id dictionaries
    idx_id_dict, id_idx_dict = get_doc_dicts(bow_df)

    # sel all courses list
    all_courses = set(course_df['COURSE_ID'])

    for user_id in test_user_ids:
        # For each user, call generate_recommendations_for_one_user() to generate the recommendation results

        # set enrolled_course_ids
        enrolled_course_ids = test_users[test_users['user'] == user_id]['item'].to_list()

        # set unselected course ids
        unselected_course_ids = all_courses.difference(enrolled_course_ids)

        # call generate_recommendations_for_one_user()
        res = generate_recommendations_for_one_user(enrolled_course_ids, unselected_course_ids, id_idx_dict, sim_matrix, threshold)

        # set users list
        for i in range(0, len(res)):
            users.append(user_id)
        # Save the result to courses, sim_scores list
        courses.extend(list(res.keys()))
        sim_scores.extend(list(res.values()))
        pass
    return users, courses, sim_scores

# sim_matrix = sim_df.to_numpy()
# threshold = 0.5
# res = generate_recommendations_for_all(threshold)
# print(res)
