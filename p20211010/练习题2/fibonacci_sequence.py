numbers=[1,1]
for i in range(20):
    numbers.append(numbers[len(numbers)-1]+numbers[len(numbers)-2])
print(numbers)