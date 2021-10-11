class Student():
    def __init__(self, name, sex, age, adr):
        self.name = name
        self.sex = sex
        self.age = age
        self.adr = adr
    def display(self):
        print(f'姓名: {self.name}', end='\n')
        print(f'性别: {self.sex}', end='\n')
        print(f'年龄: {self.age}', end='\n')
        print(f'地址: {self.adr}', end='\n')
if __name__ == '__main__':
    stu = Student('杨霖萱', '男', 20, '四川省渠县')
    stu.display()