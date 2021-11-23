from torchvision import datasets,transforms
from torch.utils.data import DataLoader

train_data = datasets.MNIST("MNIST",train=True,transform=transforms.ToTensor(),download=True)
test_data = datasets.MNIST("MNIST",train=False,download=True)

# print(train_data.data[0])

train_loader = DataLoader(train_data,batch_size=1,shuffle=True)
test_loader = DataLoader(test_data,batch_size=10,shuffle=True)

for i,(img,label) in enumerate(train_loader):
    print(i)
    # unloader = transforms.ToPILImage()
    # img = unloader(img[0])
    print(img)
    print(label)
    exit()
    # img.show()