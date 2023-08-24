from itertools import chain

face_simp = []

def range_maker(start, end):
   simp = []

   for x in range(start, end + 1, 1):
       simp.append(x)

       if x == end:
           return simp
       
face_simp.append(range_maker(0, 9))
face_simp.append(range_maker(10, 140))
face_simp = list(chain(*face_simp))
print(face_simp)
