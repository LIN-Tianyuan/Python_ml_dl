"""
Item-based collaborative filtering
"""
import math

import pandas as pd

class ItemBasedCF:
    # initialization
    def __init__(self, data_file):
        self.data_file = data_file
        self.read_data()

    # retrieve data
    def read_data(self):
        self.data = {}
        # {user:{movie1:score1, movie2:score2}}
        for line in open(self.data_file):
            user, item, score, time = line.strip().split(',')
            self.data.setdefault(user, {})
            self.data[user][item] = int(score)

    # Calculating similarity
    def item_similarity(self):
        N = dict()  # Number of times the item was purchased
        C = dict()  # Object-object co-occurrence matrix
        # {Movie1:{Movie2:number of times, Movie32:number of times}}
        # Iterate through the data to get the items where the user has performed the behavior
        for user, items in self.data.items():
            # Iterate over each item
            for i in items.keys():
                N.setdefault(i, 0)  # {Movie1: 0}
                N[i] += 1   # {Movie1: 1}
                # Object-object co-occurrence matrix
                C.setdefault(i, {})
                for j in items.keys():
                    if i == j:
                        continue
                    # Number of records
                    C[i].setdefault(j, 0)
                    C[i][j] += 1

        # Calculating similarity
        # {Movie1:{Movie2:similarity, Movie3:similarity}}
        self.W = dict()

        for i, related_item in C.items():
            self.W.setdefault(i, {})
            # Iterate over the number of occurrences common to other commodities
            for j, cij in related_item.items():
                self.W[i][j] = cij / math.sqrt(N[i] * N[j])

        return self.W


    # Recommend
    def Recommend(self, user, K=3, N=5):
        # Get the item that the user has triggered the behavior
        # {User:{Movie1: Score, Movie2: Score}}
        action_item = self.data[user]

        rank = dict()   # Used to store recommended values
        # Iterate over each behavioral commodity
        for item, score in action_item.items():
            # {Movie1:{Movie2:similarity, Movie3:similarity}}
            # ((Movie2:similarity),(),())
            for j, wj in sorted(self.W[item].items(), key=lambda  x:x[1], reverse=True)[:K]:
                # If the current item has seen
                if j in action_item.keys():
                    continue
                # Calculate recommended values
                # {Movie1: recommended value}
                rank.setdefault(j, 0)
                rank[j] += score * wj
        # Sort by recommendation
        # ((Movie, recommended value))
        return dict(sorted(rank.items(), key=lambda x:x[1], reverse=True)[:N])

if __name__ == '__main__':
    # data = pd.read_csv('../data_test/movielens电影数据/ratings.dat', sep='::', header=None, engine='python')
    # print(data)
    """
                    0     1  2          3
    0           1  1193  5  978300760
    1           1   661  3  978302109
    2           1   914  3  978301968
    3           1  3408  4  978300275
    4           1  2355  5  978824291
    ...       ...   ... ..        ...
    1000204  6040  1091  1  956716541
    1000205  6040  1094  5  956704887
    1000206  6040   562  5  956704746
    1000207  6040  1096  4  956715648
    1000208  6040  1097  4  956715569
    
    [1000209 rows x 4 columns]
    """
    # data = data.head(1000)
    # data.to_csv('./data.csv',
    #             header=None,
    #             index=False)
    icf = ItemBasedCF('./data.csv')
    icf.item_similarity()
    print(icf.Recommend('3'))
    """
    {'2916': 6.0, '39': 5.65685424949238, '2268': 5.515986323710905, '1213': 5.515986323710905, '2762': 4.330127018922194}
    """
