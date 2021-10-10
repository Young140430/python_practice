alist = [{'name': 'a', 'age': 20}, {'name': 'b', 'age': 30}, {'name': 'c', 'age': 25}]
for i in range(len(alist)):
    for j in range(i):
        if alist[i]['age'] > alist[j]['age']:
            alist[i]['age'], alist[j]['age'] = alist[j]['age'], alist[i]['age']
print(alist)