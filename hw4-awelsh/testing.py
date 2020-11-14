import math
n =27
print(math.log(n,3))
if n <= 0:
    print("False")
if math.pow(3,int(math.log(n,3))) == n:
    print("True")
else:
    print("False")
