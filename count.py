
import os



path = "./LRS2_preprocess/main"
list1 = os.listdir(path)
num = 0
for l in list1:
    path1 = os.path.join(path, l)
    num += len(os.listdir(path1))
print(num)




