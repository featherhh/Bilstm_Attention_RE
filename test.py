# a = []
# b = [word for word in range(5) if word not in a ]
# print(b)

# a = "你好 你 真 好 看"
# print(a.split(' ', maxsplit=3))

# b = "《今晚会在哪里醒来》是黄家强的一首粤语歌曲，由何启弘作词，黄家强作曲编曲并演唱"
# print(list(b))

from itertools import chain
# list1 = [1, 2, 3]
# list2 = [2, 3, 4]
# a = chain(list1, list2)
# print(list(a))
# list3 = [2, 3, 4]
# list3.extend(list1)
# print(list3)
#
# list1 = [[1,2, 3], [2, 3, 4]]
# b = chain(*list1)
# print(list(b))
#
# a = [1,2 , 3]
# b = [0]
# a.extend(b*4)
# print(a)
import torch
a = torch.load("./save_model/gz_ba_model.pth", map_location='cpu')
dict1 = {key: value.to(torch.bfloat16) for key, value in a.items()}
for key,value in dict1.items():
    print(value.dtype)