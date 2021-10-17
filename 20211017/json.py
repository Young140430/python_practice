import json5
data={
    'name':'young',
    'sex':'男',
    'major':'计算机科学与技术'
}
json_str=json5.dumps(data)
str=json5.loads(json_str)
print(data)
print(json_str)
print(str)