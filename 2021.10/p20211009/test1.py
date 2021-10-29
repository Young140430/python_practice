list=[1,13,25,17,9,11]
def my_sort(my_list):
    for i in range(len(my_list)-1):
        for j in range(i+1,len(my_list)):
            if my_list[i]>my_list[j]:
                temp=my_list[i]
                my_list[i]=my_list[j]
                my_list[j]=temp
    print(my_list)
my_sort(list)