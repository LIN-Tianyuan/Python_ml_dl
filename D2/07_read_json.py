"""
Reading and writing json files
"""
import pandas as pd

data = pd.read_json('../data_test/ratings.json',
                    orient='columns')
print(data)
"""
                  John Carson  Michelle Peterson  ...  Alex Roberts  Michael Henry
Inception                 2.5                3.0  ...           3.0            NaN
Pulp Fiction              3.5                3.5  ...           4.0            4.5
Anger Management          3.0                1.5  ...           NaN            NaN
Fracture                  3.5                5.0  ...           5.0            4.0
Serendipity               2.5                3.5  ...           3.5            1.0
Jerry Maguire             3.0                3.0  ...           3.0            NaN

[6 rows x 7 columns]
"""


data = {'Name':['Tom', 'Jack'],'Age':[28,34]}
df = pd.DataFrame(data, index=['s1','s2'])
print(df.to_json(orient='records'))  # [{"Name":"Tom","Age":28},{"Name":"Jack","Age":34}]
print(df.to_json(orient='index'))    # {"s1":{"Name":"Tom","Age":28},"s2":{"Name":"Jack","Age":34}}
print(df.to_json(orient='columns'))  # {"Name":{"s1":"Tom","s2":"Jack"},"Age":{"s1":28,"s2":34}}
print(df.to_json(orient='values'))   # [["Tom",28],["Jack",34]]

# pip install openpyxl
data = pd.read_excel("/Users/citron/Documents/GitHub/Python_ml_dl/data_test/电信用户流失数据/CustomerSurvival.xlsx")
print(data)
"""
        ID  套餐金额       额外通话时长        额外流量  改变行为  服务合约  关联购买  集团用户  使用月数  流失用户
0        1     1   792.833333  -10.450067     0     0     0     0    25     0
1        2     1   121.666667  -21.141117     0     0     0     0    25     0
2        3     1   -30.000000  -25.655273     0     0     0     0     2     1
3        4     1   241.500000 -288.341254     0     1     0     1    25     0
4        5     1  1629.666667  -23.655505     0     0     0     1    25     0
...    ...   ...          ...         ...   ...   ...   ...   ...   ...   ...
4970  4971     1  1109.333333   49.843215     0     1     0     1    25     0
4971  4972     1   197.833333  -34.987142     0     1     0     0    21     1
4972  4973     1   162.833333   71.369162     0     1     0     0    25     0
4973  4974     1   358.166667   26.315733     0     1     0     0    21     1
4974  4975     3  -655.666667  -96.435175     0     1     0     1    25     0

[4975 rows x 10 columns]
"""