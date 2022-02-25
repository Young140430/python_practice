
IMG_HEIGHT = 416
IMG_WIDTH = 416

CLASS_NUM = 20

# ANCHORS_GROUP = {
#     13: [[116,90], [156,198], [373,326]],
#     26: [[30,61], [62,45], [59,119]],
#     52: [[10,13], [16,30], [33,23]]
# }
ANCHORS_GROUP = {
    13: [[234,172], [181,316], [350,359]],
    26: [[54,154], [96,97], [104,227]],
    52: [[16,28], [49,42], [29,80]]
}

ANCHORS_GROUP_AREA = {
    13: [x * y for x, y in ANCHORS_GROUP[13]],
    26: [x * y for x, y in ANCHORS_GROUP[26]],
    52: [x * y for x, y in ANCHORS_GROUP[52]],
}
