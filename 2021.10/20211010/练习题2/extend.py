class Animal():
    def __init__(self, name, age, sex):
        self.name = name
        self.age = age
        self.sex = sex
    def display(self):
        print(f'大家好，我是{self.name}, 我今年{self.age}岁了，我是{self.sex}的。')
class Cat(Animal):
    def __init__(self, name, age, sex, func):
        super(Cat, self).__init__(name, age, sex)
        self.func = func
    def display(self):
        print('子类重构，覆盖了父类的方法。')
    def show_me(self):
        print(f'大家好，我是{self.name}, 我今年{self.age}岁了，我是{self.sex}的，我会{self.func}。')
if __name__ == '__main__':
    cat = Cat('大猫', 10, '公', '抓老鼠')
    cat.display()
    cat.show_me()