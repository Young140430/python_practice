from net import facenet
from PIL import Image
from data import tf
from torch.nn.functional import normalize
import torch

# 测试
def match(face1,face2):
    face1=normalize(face1)
    face2=normalize(face2)
    cos=torch.matmul(face1,face2.T)
    return cos

if __name__ == '__main__':
    test=facenet().cuda()
    test.load_state_dict(torch.load("param/net_last_new.pt"))
    test.eval()
    face1=tf(Image.open(r"F:\face_recognition\database\2\116.jpg")).cuda()
    print(face1.size())

    feat1=test.encode(face1[None,...])
    print(feat1.size())
    # print(feat1)
    face2=tf(Image.open(r"F:\face_recognition\database\3\104.jpg")).cuda()
    feat2=test.encode(face2[None,...])

    acc=match(feat1,feat2)
    print(acc)


# def match(face1,face2):
#     face1=normalize(face1,dim=1)
#     face2=normalize(face2,dim=1)
#     cos=torch.matmul(face1,face2)
#     return torch.max(cos)
#
# if __name__ == '__main__':
#     test=facenet().cuda()
#     test.load_state_dict(torch.load("param/net_last_new.pt"))
#     test.eval()
#
#     # s=Image.open(r"F:\face_recognition\database\1\0.jpg")
#     # s.show()
#     face1=tf(Image.open(r"F:\face_recognition\database\1\0.jpg")).cuda()
#     feat1=test.encode(face1[None,...])
#
#     database_path = r"F:\face_database\all_feat_new.pt"
#     all_feat = (torch.load(database_path)).cuda()
#
#     print(feat1)
#     print(all_feat[1][0])
    # for i in all_feat:
    #     for j in i:
    #         # print(j.size())
    #         # print(feat1.size())
    #
    #         if(torch.sum(torch.eq(feat1[0],j).float())==0):
    #             print("right")
    # acc=match(feat1,all_feat)
    # print(acc)
