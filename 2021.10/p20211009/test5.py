class alarm:
    def if_rang(self,ifrang):
        if ifrang==True:
            ring = "响了"
        else:
            ring = "关了"
        print("闹钟"+ring)
class myself:
    def __init__(self):
        myself.if_wokeup(self)
    def if_wokeup(self,name="我"):
        ifrang=True
        alarm.if_rang(self,ifrang)
        wokeup="被闹钟吵醒了"
        print(name+wokeup)
        myself.turn_off_alarm(self)
    def turn_off_alarm(self,name="我"):
        turn_off="关了闹钟"
        print(name+turn_off)
        ifrang=False
        alarm.if_rang(self,ifrang)
        print(name+"接着睡...")
m=myself()