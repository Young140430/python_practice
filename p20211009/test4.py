class person:
    def open_door(self,door):
        door.opened()

class door:
    def opened(self):
        print("门已经打开了")

p1=person()
d1=door()
p1.open_door(d1)