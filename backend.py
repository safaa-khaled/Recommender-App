import clustering_model as clustering
import courses_similarity_model as courses_similarity
import user_profile_model as user_profile
import pandas as pd

models = ("Course Similarity",
          "User Profile",
          "Clustering",
          "Clustering with PCA",
          "KNN",
          "NMF",
          "Neural Network",
          "Regression with Embedding Features",
          "Classification with Embedding Features")


def train(model_name, params):
    if model_name == models[2]:
        global model
        model = clustering.train_kmeans(clustering.features,params)
        return model
    elif model_name == models[1]:
        pass
    elif model_name == models[0]:
        pass


def predict(model_name, params):
    users = []
    courses = []
    scores = []
    res_dict = {}
    if model_name == models[2]:
        labels = model.labels_
        cluster_df = clustering.combine_cluster_labels(clustering.user_ids, labels)
        users, courses, scores= clustering.predict_kmeans(cluster_df)
    elif model_name == models[1]:
        threshold = params['sim_threshold']
        users, courses, scores = user_profile.generate_recommendation_scores(threshold)
    elif model_name == models[0]:
        threshold = round(params['sim_threshold']/100,2)
        users, courses, scores= courses_similarity.generate_recommendations_for_all(threshold)

    res_dict['USER'] = users
    res_dict['COURSE_ID'] = courses
    res_dict['SCORE'] = scores
    res_df = pd.DataFrame(res_dict, columns=['USER', 'COURSE_ID', 'SCORE'])
    return res_df
