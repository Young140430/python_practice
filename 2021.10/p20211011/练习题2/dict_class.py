class DictClass():
    def __init__(self, dic: dict):
        self.dic = dic
    def del_dict(self, key):
        del self.dic[key]
    def get_dict(self, key):
        if key in self.dic.keys():
            return self.dic[key]
        else:
            return 'not found'
    def get_key(self):
        return self.dic.keys()
    def update_dict(self, merge_dic: dict):
        self.dic.update(merge_dic)
        return list(self.dic.values()).extend(list(merge_dic.values()))
if __name__ == '__main__':
    dict1 = DictClass({'k1': 'v1', 'k2': 'v2'})
    print("t_dict.dic    :", dict1.dic)
    dict1.del_dict('k2')
    print("del_dict('k2'):", dict1.dic)
    print("get_dict('k1'):", dict1.get_dict('k1'))
    print("get_key()     :", dict1.get_key())
    dict1.update_dict({'k3': 'v3', 'k4': 'v4'})
    print("update_dict   :", dict1.dic)