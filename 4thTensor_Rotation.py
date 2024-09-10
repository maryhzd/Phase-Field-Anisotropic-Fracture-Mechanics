import numpy as np
import math
C_p = np.zeros((2, 2, 2, 2))
# C tensor components for a rock with vertical layers
C_p[0, 0, 0, 0] = 17360000000.0 #c1111 from file
C_p[1, 1, 1, 1] = 19288888888.88889 #c2222 from file

C_p[0, 0, 1, 1] = 5786666666.666666 #c1122 from file
C_p[1, 1, 0, 0] = 5786666666.666666 #c1122 from file

C_p[0, 1, 0, 1] =  5786666666 #c1212 from file
C_p[1, 0, 0, 1] = 5786666666 #c1212 from file
C_p[0, 1, 1, 0] = 5786666666 #c1212 from file
C_p[1, 0, 1, 0] = 5786666666 #c1212 from file

degrees = -15 # To calculate the C tensor components for layers oriented in 75 degrees with e1 axis
alpha = math.radians(degrees)
A = np.zeros((2, 2))
A[0,0] = math.cos(alpha )
A[0,1] = -1*math.sin(alpha )
A[1,0] = math.sin(alpha )
A[1,1] = math.cos(alpha )
C = np.zeros((2, 2, 2, 2))

for i in range(2):
    for j in range(2):
        for k in range(2):
            for l in range(2):
                for p in range(2):
                    for q in range(2):
                        for r in range(2):
                            for s in range(2):
                                C[i, j, k, l] = C[i, j, k, l] + \
                                    A[i,p]*A[j,q] *A[k,r]*A[l,s]* C_p[p,q,r,s] 

print("C1111= ", C[0, 0, 0, 0])
print("C2222= ", C[1,1,1,1])
print("C1122= ", C[0,0,1,1])
print("C1212= ", C[0,1,0,1])
print("C1112= ", C[0,0,0,1])
print("C2212= ", C[1,1,0,1])

