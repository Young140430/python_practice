numbers=[1,2]
for i in range(19):
    numbers.append(numbers[len(numbers)-1]+numbers[len(numbers)-2])
print(numbers)
sum=0
for i in range(20):
    n=numbers[i+1]/numbers[i]
    sum+=n
print(sum)