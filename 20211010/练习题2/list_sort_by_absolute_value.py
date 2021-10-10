list = [-2, 1, 3, -6]
for i in range(len(list)):
    for j in range(i):
        if abs(list[i]) < abs(list[j]):
            list[i], list[j] = list[j], list[i]
print(list)