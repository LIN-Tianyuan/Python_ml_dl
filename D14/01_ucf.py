"""
User-based collaborative filtering
"""
import numpy as np
import pandas as pd

ratings = pd.read_json('../data_test/ratings.json')
# print(ratings)
"""
                  John Carson  Michelle Peterson  ...  Alex Roberts  Michael Henry
Inception                 2.5                3.0  ...           3.0            NaN
Pulp Fiction              3.5                3.5  ...           4.0            4.5
Anger Management          3.0                1.5  ...           NaN            NaN
Fracture                  3.5                5.0  ...           5.0            4.0
Serendipity               2.5                3.5  ...           3.5            1.0
Jerry Maguire             3.0                3.0  ...           3.0            NaN
"""

login_user = "Michael Henry"

# Pearson correlation coefficient
sim_mat = ratings.corr()
# print(sim_mat)
"""
                   John Carson  Michelle Peterson  ...  Alex Roberts  Michael Henry
John Carson           1.000000           0.396059  ...      0.747018       0.991241
Michelle Peterson     0.396059           1.000000  ...      0.963796       0.381246
William Reynolds      0.404520           0.204598  ...      0.134840      -1.000000
Jillian Hobart        0.566947           0.314970  ...      0.028571       0.893405
Melissa Jones         0.594089           0.411765  ...      0.211289       0.924473
Alex Roberts          0.747018           0.963796  ...      1.000000       0.662849
Michael Henry         0.991241           0.381246  ...      0.662849       1.000000
"""
sim_scores = sim_mat.loc[login_user]
# Remove negatively correlated data(A strong correlation)
sim_scores = sim_scores[sim_scores > 0.6]
# Get rid of himself
sim_scores = sim_scores.drop(login_user)
# print(sim_scores)

rec_movie = {}
# {'Movie1': [Score1, Score2], [Similarity1， Similarity2]}
for sim_user, sim_score in sim_scores.items():
    # Traverse what movies sim_user has watched
    sim_movies = ratings[sim_user].dropna()
    # Determine the movies watched by similar users and whether the logged-in user has watched them
    for m, s in sim_movies.items():
        if np.isnan(ratings[login_user][m]):
            # m: This movie has not been watched by logged-in users
            if m not in rec_movie.keys():
                rec_movie[m] = [[], []]
            rec_movie[m][0].append(s)
            rec_movie[m][1].append(sim_score)

# print(rec_movie)
"""
{'Inception': [[2.5, 3, 3.0], [0.9912407071619305, 0.924473451641905, 0.6628489803598702]], 
'Anger Management': [[3.0, 3.0, 2], [0.9912407071619305, 0.8934051474415642, 0.924473451641905]], 
'Jerry Maguire': [[3.0, 4.5, 3, 3.0], [0.9912407071619305, 0.8934051474415642, 0.924473451641905, 0.6628489803598702]]}
"""

rec_res = {}
for m, val in rec_movie.items():
    val = np.array(val)
    score = (val[0] * val[1]).sum()
    rec_res[m] = score

rec_res = sorted(rec_res.items(), key=lambda x:x[1], reverse=True)
print(rec_res)
"""
[('Jerry Maguire', 11.756012580978155), 
('Anger Management', 7.502884467094294), 
('Inception', 7.240069063910152)]
"""