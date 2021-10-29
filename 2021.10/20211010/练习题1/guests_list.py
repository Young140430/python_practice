guests=['李智宇','刘泉','马涛']
for guest in guests:
    print(guest+'，你好！我想邀请你一起共进晚餐。')
print(guests[2]+'因为临时有事不能赴约。')
guests[2]='李顺'
for guest in guests:
    print(guest+'，你好！我想邀请你一起共进晚餐。')
print('我找到了一个更大的餐桌，可以再邀请3个人共进晚餐。')
guests.insert(0,'张银萍')
guests.insert(2,'罗艺')
guests.append('曾小龙')
print('我一共邀请了'+str(len(guests))+'人一起共进晚餐')
'''for guest in guests:
    print(guest+'，你好！我想邀请你一起共进晚餐。')
print('我刚得知新购买的餐桌无法及时送达，因此只能邀请两位嘉宾。')
while len(guests)>2:
    print(guests[len(guests)-1]+'，你好！我很抱歉，因为临时有事无法邀请你来共进晚餐。')
    guests.pop()
for guest in guests:
    print(guest+'，你好！我想邀请你一起共进晚餐。')
del guests[1]
del guests[0]
print('嘉宾列表：',guests)
'''