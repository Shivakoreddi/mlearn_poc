import numpy as np

import pandas as pd

import pandas as pd


np.random.seed(42)

n = 100  # increase this for deeper forests


data = pd.DataFrame({
    'Department': np.random.choice(['ICU', 'SURGERY', 'ORTHO', 'PEDIA', 'CARDIO'], size=n),
    'Bed_Occupancy': np.random.randint(0, 100, size=n),
    'Emergency_Admissions': np.random.randint(0, 50, size=n),
    'Weekend': np.random.choice([0, 1], size=n),
})

print(data)

##generate target based on some logic

data['beds_required'] = ((data['Department']=='ICU') & (data['Emergency_Admissions']>10)).astype(int)
print(data)
 ## forest requirements:-
 ##1. we need features at least 2 or more, becoz to involve multiple trees ,we need different features to evaluate
 ##2. each tree will read each feature with its threshold
 ##3.build forest
 ##4. first able to simulate each tree with its parametrs - 1. left_mask/right_mask, threshold calculation,x_left,y_left/x_rightmy_right
 ##5. nl,nR/weighted_gini,gini_left,gini_right
 ##6. now loop these parametrs for each feature
y = 'beds_required'

# Gini Impurity function
def gini_impurity(y_group):
    m = len(y_group)
    if m == 0:
        return 0
    p1 = np.sum(y_group == 0,axis=None) / m
    p2 = np.sum(y_group == 1,axis=None) / m
    return 1 - p1**2 - p2**2

def mask_calc(left_mask,right_mask):
    x_left, y_left = data[left_mask], data[left_mask]
    x_right, y_right = data[right_mask], data[right_mask]
    print(x_left)
    print(y_left)
    print(x_right)
    print(y_right)
    nl = len(x_left)
    print(nl)
    nr = len(x_right)
    print(nr)
    # Calculate impurities
    gini_left = gini_impurity(y_left)
    gini_right = gini_impurity(y_right)
    # Weighted Gini
    weighted_gini = (nl / n) * gini_left + (nr / n) * gini_right
    return weighted_gini


## loop for all features to get weighted_gini
for x in data.columns:
     print(x)
     if x == 'Department':
         print(x)
         left_mask=data[x]=='ICU'
         right_mask=data[x]!='ICU'
         print(left_mask)
         print(right_mask)
         print("Weighted Gini:", mask_calc(left_mask, right_mask))


         # # Output
         # print("\nGini Left:", gini_left)
         # print("Gini Right:", gini_right)




#
# print(left_mask)
# print(right_mask)
# x_left, y_left = x[left_mask], y[left_mask]
# x_right, y_right = x[right_mask], y[right_mask]



