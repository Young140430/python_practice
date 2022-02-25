import torchvision

net=torchvision.models.densenet121(True)
print(net)
net.classifier.out_features=2
print(net.classifier)