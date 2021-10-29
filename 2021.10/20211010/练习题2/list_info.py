class Listinfo():
    def __init__(self, l: list):
        self.l = l
    def add_key(self, keyname: str or int):
        self.l.append(keyname)
    def get_key(self, num: int):
        return self.l[num]
    def update_list(self, l2: list):
        self.l.extend(l2)
    def del_key(self):
        self.l.pop()
if __name__ == '__main__':
    list_info = Listinfo([1, 3, 15, 9, 17, 'y', 'l'])
    print(list_info.l)
    list_info.add_key('add_key')
    print(list_info.l)
    print(list_info.get_key(4))
    list_info.update_list([False, 'x'])
    print(list_info.l)
    list_info.del_key()
    print(list_info.l)