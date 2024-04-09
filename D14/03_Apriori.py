"""
Association rule:Apriori
"""


def load_data_set():
    with open('item.txt', 'r') as f:
        data = f.readlines()
        res = []
        for i in data:
            res.append(i.strip().split(','))
        return res


def createC1(dataSet):
    # Iterate over each item for each purchase record
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            # If the product exists, different additions
            if [item] not in C1:
                C1.append([item])
    C1.sort()
    return list(map(frozenset, C1))


def scanD(D, Ck, min_support):
    ssCnt = {}  # {frozenset{'bread': times}
    for tid in D: # [{bread, cake, milk, tea}, {}, {}]
        for can in Ck: # [{bread}, {milk}, {tea}]
            if can.issubset(tid):
                if can not in ssCnt:
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1
    # Calculation of support: number of occurrences of commodity combinations / total number of samples
    numItems = float(len(D))

    retList = []    # Save frequent itemsets
    supportData = {}    # Record the level of support for each item
    # Iterate over the number of occurrences of each item in the candidate set
    for key in ssCnt:
        support = ssCnt[key] / numItems
        # filtration
        if support >= min_support:
            retList.insert(0, key)
        supportData[key] = support
    return retList, supportData


def aprioriGen(Lk, k):
    """
    # Generate a candidate set for the next itemset based on the frequent itemsets of the previous itemset
    :param Lk: The set of frequent terms of the previous term
    :param k:  Number of items to generate
    :return:   Candidate set for the current itemset
    """
    retList = []
    for i in range(len(Lk) - 1):
        for j in range(i+1, len(Lk)):
            if len(Lk[i] & Lk[j]) == (k - 2):
                retList.append(Lk[i] | Lk[j])
    retList = list(set(retList))
    return retList


def apriori(dataSet, min_support):
    # Master Functions for Overall Logic
    # Generate a candidate set of 1-item sets
    C1 = createC1(dataSet)
    # Converting a dataSet into a list set of fixed collections
    D = list(map(frozenset, dataSet))
    # Filtering for minimum support yields frequent 1-term sets
    L1, supportData = scanD(D, C1, min_support)
    L = [L1]  # Store the frequent itemsets for each itemset
    k = 2  # The number of items in the next itemset to be generated

    while len(L[k-2]) > 0:  # There are items in the current frequent itemset in order to generate a candidate set for the next itemset
        # Generate a candidate set for the next itemset based on the frequent itemsets of the previous itemset
        Ck = aprioriGen(L[k-2], k)
        # Calculate support, filter minimum support
        Lk, supK = scanD(D, Ck, min_support)
        L.append(Lk)
        supportData.update(supK)
        k += 1

    return L, supportData


if __name__ == '__main__':
    dataSet = load_data_set()
    L, supK = apriori(dataSet, 0.5)
    # The set of frequent terms for each
    # for i in L:
    #     print(i)
    """
    [frozenset({'tea'}), frozenset({'milk'}), frozenset({'bread'})]
    [frozenset({'milk', 'tea'}), frozenset({'bread', 'milk'}), frozenset({'bread', 'tea'})]
    []
    """
    for i, j in supK.items():
        print(i, ':', j)
