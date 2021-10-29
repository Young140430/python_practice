sum = 0
for i in range(1, 21):
    temp = i
    for j in range(1, i):
        temp *= j
    sum += temp
print(f'1+2!+3!+...+20!的和为{sum}')