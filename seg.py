import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader

from PIL import Image
import torch.utils.data as data
import torchvision
from torchvision import transforms, datasets


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

num_images = 10                                   #生成する画像数
length = 224                                          #画像のサイズ
length2 = 224
imgs = np.zeros([num_images, 1, length, length])     #ゼロ行列を生成,入力画像
imgs_ano = np.zeros([num_images, 1, length, length]) #出力画像
image = np.zeros([num_images, 1, length, length])   #added
image_ano = np.zeros([num_images, 1, length, length])



##################
#http://kaga100man.com/2019/01/09/post-89/
#output_map = np.zeros(h * w).reshape(h, w)
# data augmentation :    https://roshiago.blog.ss-blog.jp/2019-07-15
class MyDataset(data.Dataset):
    def __init__(self, dir_path, input_size,fl):
        super().__init__()

        self.dir_path = dir_path
        self.input_size = input_size
        #self.transform=transform

        # hymenoptera_data/{train or val}/{ants or bees}/ファイル名.jpg
        # ex. hymenoptera_data/val/bees/2478216347_535c8fe6d7.jpg
        if fl==0:
            self.image_paths = [str(p) for p in Path(self.dir_path).glob("**/*.jpeg")]
        if fl == 1:
            self.image_paths = [str(p) for p in Path(self.dir_path).glob("**/*.png")]
        self.len = len(self.image_paths)
        print(self.len)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        p = self.image_paths[index]

        # 入力
        image = np.zeros([self.len, 1, length2, length2])
        image = Image.open(p)
        #image=self.transform(image)
        image = torchvision.transforms.functional.to_grayscale(image)
        image = image.resize(self.input_size)
        image = np.array(image)/255.

        # Swap channel axis
        #     numpy image: Height x Width x Channel
        #     torch image: Channel x Height x Width
        #image = np.transpose(image, (2, 0, 1))
        image = torch.from_numpy(image)



        return image

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # Normalize(平均, 偏差)

train_dataset = MyDataset("./data/train/2/", (224, 224),0)
label_dataset = MyDataset("./data/label/2/", (224, 224),1)
#print("dddd",train_dataset.shape)
#~~~~~~~~~~~^
#hymenoptera_dataset = datasets.ImageFolder(root='./data/train/',
 #                                          )
#dataset_loader = torch.utils.data.DataLoader(hymenoptera_dataset,
 #                                            batch_size=2, shuffle=True,
 #                                            num_workers=1)
#images=next(iter(dataset_loader))
for i, data in enumerate(train_dataset, 0):
   image[i,:,:,:] = data
   torch.save(image, 'data_drive_path{}'.format(i))
print("dddd",image.shape)
#image22=image[0,0,:,:]
#plt.imshow(image22, cmap = "gray")
#plt.savefig("input_auto")
##plt.show()

for i, data in enumerate(label_dataset, 0):
   image_ano[i,:,:,:] = data



#################

imgs = torch.tensor(image, dtype = torch.float32)   #added
#imgs = torch.tensor(imgs, dtype = torch.float32)                 #ndarray - torch.tensor
print("aaa",imgs.shape)
imgs_ano = torch.tensor(image_ano, dtype = torch.float32)
#imgs_ano = torch.tensor(imgs_ano, dtype = torch.float32)           #ndarray - torch.tensor
data_set = TensorDataset(imgs, imgs_ano)
data_loader = DataLoader(data_set, batch_size = 100, shuffle = True)

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        #Encoder Layers
        self.conv1 = nn.Conv2d(in_channels = 1,
                               out_channels = 16,
                               kernel_size = 3,
                               padding = 1)
        self.conv2 = nn.Conv2d(in_channels = 16,
                               out_channels = 4,
                               kernel_size = 3,
                               padding = 1)
        #Decoder Layers
        self.t_conv1 = nn.ConvTranspose2d(in_channels = 4, out_channels = 16,
                                          kernel_size = 2, stride = 2)
        self.t_conv2 = nn.ConvTranspose2d(in_channels = 16, out_channels = 1,
                                          kernel_size = 2, stride = 2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #encode#
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        #decode#
        x = self.relu(self.t_conv1(x))
        x = self.sigmoid(self.t_conv2(x))
        return x

# ******ネットワークを選択******
net = ConvAutoencoder() #.to(device)
loss_fn = nn.MSELoss()  # 損失関数の定義
optimizer = optim.Adam(net.parameters(), lr=0.01)

losses = []  # epoch毎のlossを記録
epoch_time = 30
for epoch in range(epoch_time):
        running_loss = 0.0  # epoch毎のlossの計算
        net.train()
        for i, (XX, yy) in enumerate(data_loader):
            XX=XX #.to(device)
            yy = yy #.to(device)
            optimizer.zero_grad()
            #print(XX.shape)
            y_pred = net(XX)
            loss = loss_fn(y_pred, yy)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print("epoch:", epoch, " loss:", running_loss / (i + 1))
        losses.append(running_loss / (i + 1))

    # lossの可視化
plt.plot(losses)
plt.ylabel("loss")
plt.xlabel("epoch time")
plt.savefig("loss_auto")
plt.show()

#############################

net.eval()            #評価モード
#今まで学習していない画像を1つ生成
num_images = 1
img_test = np.zeros([num_images, 1, length, length])
imgs_test_ano = np.zeros([num_images, 1, length, length])
image_for_test = np.zeros([num_images, 1, length, length])

"""
for i in range(num_images):
    centers = []
    img = np.zeros([length, length])
    img_ano = np.zeros([length, length])
    for j in range(6):
        img, img_ano, centers = rectangle(img, img_ano, centers, 7)
    img_test[i, 0, :, :] = img
"""
test_dataset = MyDataset("./data/test/", (224, 224),0)

for i, data in enumerate(test_dataset, 0):
   #image = train_dataset
   image_for_test[i,:,:,:] = data

#img_test = img_test.reshape([1, 1, 64, 64])
print(image.shape)
img_test = image_for_test.reshape([1, 1, 224, 224])
img_test = torch.tensor(img_test, dtype = torch.float32)
#img_test=img_test.to(device)
img_test = net(img_test)             #生成した画像を学習済のネットワークへ
img_test = img_test.detach().numpy() #torch.tensor - ndarray
#img_test = img_test[0, 0, :, :]
image22=img_test[0,0,:,:]


print("size=",image22.shape)
plt.imshow(image22, cmap = "gray")  #outputデータの可視化
plt.savefig("output_auto")
plt.show()
