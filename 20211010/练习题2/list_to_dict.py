str = "k:1|k1:2|k2:3|k3:4"
elements = str.split('|')
dict = {}
for kv in elements:
    key = kv.split(':')[0]
    value = kv.split(':')[1]
    dict[key] = value
print(dict)