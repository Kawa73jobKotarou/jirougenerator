#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

import cgi
from PIL import Image, ImageDraw
import io
import os
import sys
Forms = cgi.FieldStorage()
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding = 'utf-8')        #printで出力する内容をutf-8にする
print('Content-Type: text/html; charset=UTF-8\n')   
Text = """
<!DOCTYPE html>
<html lang="ja">
    <head>
        <meta charset="UTF-8">
        <meta http-equiv="X-UA-Compatible" content="IE=edge">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <!-- シンプルCSS -->
        <link rel="stylesheet" href="https://cdn.simplecss.org/simple.css">
        <link rel="stylesheet" href="../style.css">
        <title>ラーメン二郎ジェネレーター</title>
    </head>
    <body class="anti">
        <header>
            <h1>二郎の写真も見たくない！！！</h1>
        </header>
        <main>
            <section>
                <p>変換したい二郎を入力してください</p>
                <form name="fm" action="anti.py" method="post" enctype="multipart/form-data" accept=".jpg, .jpeg, .png, .JPEG">
                    <input class="anti" type="file" id="imagefile" name="origin_image">
                    <input type="submit" value="成敗する">
                </form>
            </section>
            <section>
                <div class="flex ramen-images" id="makeImg">
                    <div>
                        <p>変換前の二郎</p>
                        <img id="orgImg" src="../%s" alt="Original Image">
                    </div>
                    <div>
                        <p>変換した画像</p>
                        <img id="rsltImg" src="../%s" alt="Result Image">
                    </div>
                </div>
                <p>%s</p>
            </section>
        </main>
        <footer class="anti">
            <a class="jirou_button" href="../index.html">二郎ジェネレータ</a>
            <p>二郎なんてもう食べない！</p>
        </footer>


        <script type="text/javascript">
            var elem = document.getElementById("orgImg");
            // console.log(elem.src);
            document.getElementById("imagefile").addEventListener("change", function(){
                var fileList = this.files ;
                // console.log(fileList);
                // console.log(fileList[0]);
                var blobUrl = window.URL.createObjectURL(fileList[0]);
                // console.log(blobUrl);
                elem.src = blobUrl;
             });
        </script>
    </body>
</html>
"""
noimage_path="img/no_image.png"
try:
    # 画像の読み込み
    origin_image = Forms['origin_image']
    # 画像データを取得
    image_data = origin_image.value
    # 画像データをPIL Imageに変換
    image = Image.open(io.BytesIO(image_data))
    # 画像のサイズを256x256に変換
    image = image.resize((256, 256))
    origin_image_path = "img/originjirou_image.jpg"
    image.save(origin_image_path)

    # ここからネットワーク


    # # 画像中心座標の計算
    # center_x, center_y = image.width // 2, image.height // 2

    # # 十字の線を描画
    # draw = ImageDraw.Draw(image)
    # line_color = (255, 0, 0)  # 十字の線の色 (赤色)
    # line_width = 2  # 十字の線の幅

    # # 横線を描画
    # draw.line([(0, center_y), (image.width, center_y)], fill=line_color, width=line_width)

    # # 縦線を描画
    # draw.line([(center_x, 0), (center_x, image.height)], fill=line_color, width=line_width)


    import os
    import itertools
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.utils.data
    import torchvision.transforms as transforms
    from torchvision.utils import make_grid
    from torch.autograd import Variable
    import random
    from PIL import Image
    import matplotlib.pyplot as plt
    import pdb
    load_size = 256 # オリジナルの画像はこのサイズにリサイズ
    fine_size = 256  # 286x286の画像からランダムに256x256をcrop
    batch_size = 2
    num_epoch = 50

    lr = 0.0002  # initial learning rate for adam
    beta1 = 0.5  # momentum term of adam
    log_dir = 'logs'

    save_epoch_freq = 1

    cuda = torch.cuda.is_available()

    class ResNetBlock(nn.Module):

        def __init__(self, dim):
            super(ResNetBlock, self).__init__()
            conv_block = []
            conv_block += [nn.ReflectionPad2d(1),
                        nn.Conv2d(dim, dim, kernel_size=3),
                        nn.InstanceNorm2d(dim),
                        nn.ReLU(True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(dim, dim, kernel_size=3),
                        nn.InstanceNorm2d(dim)]
            self.conv_block = nn.Sequential(*conv_block)

        def forward(self, x):
            out = x + self.conv_block(x)
            return out

    class Generator(nn.Module):
        
        def __init__(self):
            super(Generator, self).__init__()

            self.model = nn.Sequential(
                nn.ReflectionPad2d(3),

                nn.Conv2d(3, 64, kernel_size=7),
                nn.InstanceNorm2d(64),
                nn.ReLU(True),

                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(128),
                nn.ReLU(True),

                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(256),
                nn.ReLU(True),

                ResNetBlock(256),
                ResNetBlock(256),
                ResNetBlock(256),
                ResNetBlock(256),
                ResNetBlock(256),
                ResNetBlock(256),
                ResNetBlock(256),
                ResNetBlock(256),
                ResNetBlock(256),

                nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(128),
                nn.ReLU(True),

                nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(64),
                nn.ReLU(True),

                nn.ReflectionPad2d(3),
                nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=0),
                nn.Tanh()
            )

            # initialize weights
            self.model.apply(self._init_weights)

        def forward(self, input):
            return self.model(input)

        def _init_weights(self, m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                nn.init.normal_(m.weight.data, 0.0, 0.02)

    class Discriminator(nn.Module):
        
        def __init__(self):
            super(Discriminator, self).__init__()

            self.model = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm2d(128),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm2d(256),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
                nn.InstanceNorm2d(512),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
            )

            # initialize weights
            self.model.apply(self._init_weights)

        def forward(self, input):
            return self.model(input)

        def _init_weights(self, m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                nn.init.normal_(m.weight.data, 0.0, 0.02)

    class ImagePool():

        def __init__(self, pool_size):
            self.pool_size = pool_size
            if self.pool_size > 0:
                self.num_imgs = 0
                self.images = []

        def query(self, images):
            # プールを使わないときはそのまま返す
            if self.pool_size == 0:
                return Variable(images)
            return_images = []
            for image in images:
                # バッチの次元を削除して3Dテンソルに
                image = torch.unsqueeze(image, 0)
                if self.num_imgs < self.pool_size:
                    self.num_imgs = self.num_imgs + 1
                    self.images.append(image)
                    return_images.append(image)
                else:
                    p = random.uniform(0, 1)
                    if p > 0.5:
                        random_id = random.randint(0, self.pool_size - 1)
                        tmp = self.images[random_id].clone()
                        self.images[random_id] = image
                        return_images.append(tmp)
                    else:
                        return_images.append(image)
            return_images = Variable(torch.cat(return_images, 0))
            return return_images
        
    class GANLoss(nn.Module):
        
        def __init__(self):
            super(GANLoss, self).__init__()
            self.real_label_var = None
            self.fake_label_var = None
            self.loss = nn.MSELoss()
        
        def get_target_tensor(self, input, target_is_real):
            target_tensor = None
            if target_is_real:

                create_label = ((self.real_label_var is None) or (self.real_label_var.numel() != input.numel()))
                if create_label:
                    real_tensor = torch.ones(input.size())
                    if cuda:
                        real_tensor = real_tensor.cuda()
                    self.real_label_var = Variable(real_tensor, requires_grad=False)
                target_tensor = self.real_label_var
            else:
                create_label = ((self.fake_label_var is None) or (self.fake_label_var.numel() != input.numel()))
                if create_label:
                    fake_tensor = torch.zeros(input.size())
                    if cuda:
                        fake_tensor = fake_tensor.cuda()
                    self.fake_label_var = Variable(fake_tensor, requires_grad=False)
                target_tensor = self.fake_label_var
            return target_tensor

        def __call__(self, input, target_is_real):
            target_tensor = self.get_target_tensor(input, target_is_real)
            return self.loss(input, target_tensor)

    class CycleGAN(object):
        
        def __init__(self, log_dir='logs'):
            self.netG_A = Generator()
            self.netG_B = Generator()
            self.netD_A = Discriminator()
            self.netD_B = Discriminator()

            if cuda:
                self.netG_A.cuda()
                self.netG_B.cuda()
                self.netD_A.cuda()
                self.netD_B.cuda()

            self.fake_A_pool = ImagePool(50)
            self.fake_B_pool = ImagePool(50)

            # targetが本物か偽物かで代わるのでオリジナルのGANLossクラスを作成
            self.criterionGAN = GANLoss()
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()

            # Generatorは2つのパラメータを同時に更新
            self.optimizer_G = torch.optim.Adam(
                itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                lr=lr,
                betas=(beta1, 0.999))
            self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=lr, betas=(beta1, 0.999))
            self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=lr, betas=(beta1, 0.999))
            self.optimizers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D_A)
            self.optimizers.append(self.optimizer_D_B)
            
            self.log_dir = log_dir
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
        
        def set_input(self, input):
            input_A = input['A']
            input_B = input['B']
            if cuda:
                input_A = input_A.cuda()
                input_B = input_B.cuda()
            self.input_A = input_A
            self.input_B = input_B
            self.image_paths = input['path_A']

        def backward_G(self, real_A, real_B):
            # Generatorに関連するlossと勾配計算処理
            lambda_idt = 0.5
            lambda_A = 10.0
            lambda_B = 10.0

            # G_A, G_Bは変換先ドメインの本物画像を入力したときはそのまま出力するべき
            # netG_AはドメインAの画像からドメインBの画像を生成するGeneratorだが
            # ドメインBの画像も入れることができる
            # その場合は何も変換してほしくないという制約
            # TODO: idt_Aの命名はよくない気がする idt_Bの方が適切では？
            idt_A = self.netG_A(real_B)
            loss_idt_A = self.criterionIdt(idt_A, real_B) * lambda_B * lambda_idt

            idt_B = self.netG_B(real_A)
            loss_idt_B = self.criterionIdt(idt_B, real_A) * lambda_A * lambda_idt

            # GAN loss D_A(G_A(A))
            # G_Aとしては生成した偽物画像が本物（True）とみなしてほしい
            fake_B = self.netG_A(real_A)
            pred_fake = self.netD_A(fake_B)
            loss_G_A = self.criterionGAN(pred_fake, True)

            # GAN loss D_B(G_B(B))
            # G_Bとしては生成した偽物画像が本物（True）とみなしてほしい
            fake_A = self.netG_B(real_B)
            pred_fake = self.netD_B(fake_A)
            loss_G_B = self.criterionGAN(pred_fake, True)
            
            # forward cycle loss
            # real_A => fake_B => rec_Aが元のreal_Aに近いほどよい
            rec_A = self.netG_B(fake_B)
            loss_cycle_A = self.criterionCycle(rec_A, real_A) * lambda_A
            
            # backward cycle loss
            # real_B => fake_A => rec_Bが元のreal_Bに近いほどよい
            rec_B = self.netG_A(fake_A)
            loss_cycle_B = self.criterionCycle(rec_B, real_B) * lambda_B


    #新たなlossを追加

            
            # combined loss
            #loss_G = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B
            loss_G = loss_G_A + loss_G_B + loss_idt_A + loss_idt_B #cyclelossを使わないばーじょん

            loss_G.backward()

            # 次のDiscriminatorの更新でfake画像が必要なので一緒に返す
            #return loss_G_A.data[0], loss_G_B.data[0], loss_cycle_A.data[0], loss_cycle_B.data[0], loss_idt_A.data[0], loss_idt_B.data[0], fake_A.data, fake_B.data
            return loss_G_A.item(), loss_G_B.item(), loss_cycle_A.item(), loss_cycle_B.item(), loss_idt_A.item(), loss_idt_B.item(), fake_A.data, fake_B.data

        def backward_D_A(self, real_B, fake_B):
            # ドメインAから生成したfake_Bが本物か偽物か見分ける

            # TODO: これは何をしている？
            # fake_Bを直接使わずに過去に生成した偽画像から新しくランダムサンプリングしている？
            fake_B = self.fake_B_pool.query(fake_B)

            # 本物画像を入れたときは本物と認識するほうがよい
            pred_real = self.netD_A(real_B)
            loss_D_real = self.criterionGAN(pred_real, True)

            # ドメインAから生成した偽物画像を入れたときは偽物と認識するほうがよい
            # fake_Bを生成したGeneratorまで勾配が伝搬しないようにdetach()する
            pred_fake = self.netD_A(fake_B.detach())
            loss_D_fake = self.criterionGAN(pred_fake, False)

            # combined loss
            loss_D_A = (loss_D_real + loss_D_fake) * 0.5
            loss_D_A.backward()
            
            return loss_D_A.item()

        def backward_D_B(self, real_A, fake_A):
            # ドメインBから生成したfake_Aが本物か偽物か見分ける

            fake_A = self.fake_A_pool.query(fake_A)
            
            # 本物画像を入れたときは本物と認識するほうがよい
            pred_real = self.netD_B(real_A)
            loss_D_real = self.criterionGAN(pred_real, True)

            # 偽物画像を入れたときは偽物と認識するほうがよい
            pred_fake = self.netD_B(fake_A.detach())
            loss_D_fake = self.criterionGAN(pred_fake, False)
            
            # combined loss
            loss_D_B = (loss_D_real + loss_D_fake) * 0.5
            loss_D_B.backward()
            
            return loss_D_B.item()

        def optimize(self):
            real_A = Variable(self.input_A)
            real_B = Variable(self.input_B)
            
            # update Generator (G_A and G_B)
            self.optimizer_G.zero_grad()
            loss_G_A, loss_G_B, loss_cycle_A, loss_cycle_B, loss_idt_A, loss_idt_B, fake_A, fake_B = self.backward_G(real_A, real_B)
            self.optimizer_G.step()

            # update D_A
            self.optimizer_D_A.zero_grad()
            loss_D_A = self.backward_D_A(real_B, fake_B)
            self.optimizer_D_A.step()
            
            # update D_B
            self.optimizer_D_B.zero_grad()
            loss_D_B = self.backward_D_B(real_A, fake_A)
            self.optimizer_D_B.step()

            ret_loss = [loss_G_A, loss_D_A,
                        loss_G_B, loss_D_B,
                        loss_cycle_A, loss_cycle_B,
                        loss_idt_A, loss_idt_B]

            return np.array(ret_loss)

        def train(self, data_loader):
            running_loss = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            for batch_idx, data in enumerate(data_loader):
                self.set_input(data)
                losses = self.optimize()
                running_loss += losses
            running_loss /= len(data_loader)
            return running_loss
        
        def save_network(self, network, network_label, epoch_label):
            save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
            save_path = os.path.join(self.log_dir, save_filename)
            torch.save(network.state_dict(), save_path)

        def load_network(self, network, network_label, epoch_label):
            load_filename = '%s_net_%s.pth' % (epoch_label, network_label)
            load_path = os.path.join(self.log_dir, load_filename)
            load_path = 'cgi-bin/'+load_path
            network.load_state_dict(torch.load(load_path))


        def save(self, label):
            self.save_network(self.netG_A, 'G_A', label)
            self.save_network(self.netD_A, 'D_A', label)
            self.save_network(self.netG_B, 'G_B', label)
            self.save_network(self.netD_B, 'D_B', label)
        
        def load(self, label):
            self.load_network(self.netG_A, 'G_A', label)
            # self.load_network(self.netD_A, 'D_A', label)
            self.load_network(self.netG_B, 'G_B', label)
            # self.load_network(self.netD_B, 'D_B', label)

    model = CycleGAN()
    model.load('epoch50')
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    image = image.convert('RGB')

    image = transform(image).unsqueeze(0)

    # モデルに画像を渡して変換を行う
    fake_A = model.netG_A.cpu()(image)

    fake_A = (fake_A + 1) / 2  # 画像のピクセル値を[0, 1]の範囲にスケーリング

    image = fake_A[0].detach()

    image=transforms.ToPILImage()(image)

    # ここまでネットワーク

    # 出力画像を保存

    output_image_path = "img/anti_generated_image.jpg"

    image.save(output_image_path)

    txt2 = "保存しました"
    print(Text % (origin_image_path, output_image_path, txt2))

except:
    txt1 = "初期化されました"
    print(Text % (noimage_path, noimage_path, txt1))
