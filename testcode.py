# from itertools import chain

# face_simp = []

# def range_maker(start, end):
#    simp = []

#    for x in range(start, end + 1, 1):
#        simp.append(x)

#        if x == end:
#            return simp
       
# face_simp.append(range_maker(0, 9))
# face_simp.append(range_maker(10, 140))
# face_simp = list(chain(*face_simp))
# print(face_simp)




#### 


  
import pandas as pd
from numpy.random import randint
  
dict = {'Name':['Martha', 'Tim', 'Rob', 'Georgia'],
        'Maths':[87, 91, 97, 95],
        'Science':[83, 99, 84, 76]
       }
  
df = pd.DataFrame(dict)
  

  
df.loc[len(df.index)] = ['Amy', 89, 93] 
  
print(df)