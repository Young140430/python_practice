import torch
from UNet import *


transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])
img_path = r"E:\DATE\DUTS\DUTS-TE\DUTS-TE-Image"

if __name__ == '__main__':
    for img_file in glob.glob("{}/*.jpg".format(img_path)):
        image = cv2.imread(img_file)
        img_h, img_w = image.shape[0], image.shape[1]
        side_len = max(img_h, img_w)
        image_new = np.zeros((side_len, side_len, 3), dtype = np.uint8)
        image_new[:img_h, :img_w] = image
        img_h_new, img_w_new = image_new.shape[0], image_new.shape[1]
        image_new = cv2.cvtColor(image_new, cv2.COLOR_BGR2RGB)
        image_new = cv2.resize(image_new, (256,256))
        resize_w = 256 / img_w_new
        resize_h = 256 / img_h_new
        image_data = transforms(image_new)
        image_data = torch.unsqueeze(image_data, dim = 0)


        net = MainNet()
        net.load_state_dict(torch.load(r"E:\DATE\DUTS\params\last_param.pt"))
        out = net(image_data)

        # print(out.shape)
        out = torch.squeeze(out,dim = 0)
        # print(out.shape)
        out = np.array(out.detach().numpy())

        out = out.transpose(1,2,0)
        # out = cv2.cvtColor(out,cv2.COLOR_BGR2GRAY)

        out = torch.Tensor(out)
        # print(out.shape)

        out_mask = torch.ge(out,0.8).float()
        out_mask = np.array(out*255,dtype = np.uint8)
        # print(out_mask)

        # ret,binary = cv2.threshold(out,0,255,cv2.THRESH_OTSU and cv2.THRESH_BINARY)

        # print(out_mask)
        # out_mask = np.array(out_mask,dtype = np.uint8)

        # out_mask = cv2.imread(out_mask)
        cv2.imshow("out",out_mask)
        cv2.waitKey()
        cv2.destroyAllWindows()
