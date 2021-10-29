class student():
    def __init__(self,name,age,grade):
        self.name=name
        self.age=age
        self.grade=grade
    def get_name(self):
        print(self.name)
    def get_age(self):
        print(self.age)
    def get_course(self):
        print(max(self.grade[0],self.grade[1],self.grade[2]))
s1=student('Young',20,[60,90,80])
s1.get_name()
s1.get_age()
s1.get_course()