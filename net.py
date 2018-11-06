import torch.nn as nn
from torchvision import models


def weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias:
            nn.init.constant_(m.bias.data, 0)


class PixelLinkNet(nn.Module):
    def __init__(self, backbone='vgg16', pretrained=False, version='2s',
                 add_extra_block=True, extra_block_expand=2.0):
        # TODO: pretrained is not really fully supported since in that case the input should be normalized to [0,1]
        #       and then whitened with mean & var
        super(PixelLinkNet, self).__init__()
        self.version = version
        self.add_extra_block = add_extra_block
        self.pretrained = pretrained

        # blocks resolution : x2, x4, x8, x16
        if backbone == 'vgg16':
            backbone = models.vgg16(pretrained=pretrained)
            self.blocks = [backbone.features[:9],      # conv1_1, conv1_2, conv2_1, conv2_2
                           backbone.features[9:16],    # conv3_1 : conv3_3
                           backbone.features[16:23],   # conv4_1 : conv4_3
                           backbone.features[23:30]]   # conv5_1 : conv5_3
            num_features = [128, 256, 512, 512]
        elif backbone.startswith('resnet'):
            f = getattr(models, backbone)
            backbone = f(pretrained=pretrained)
            self.blocks = [nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu),
                           nn.Sequential(backbone.maxpool, backbone.layer1),
                           backbone.layer2,
                           backbone.layer3,
                           backbone.layer4]
            num_features = [64]
            if backbone == 'resnet18' or backbone == 'resnet34':
                num_features = num_features + [64, 128, 256, 512]
            else:  # resnet50 / resnet101 / resnet152
                num_features = num_features + [4*64, 4*128, 4*256, 4*512]
        self.blocks = nn.ModuleList(self.blocks)

        if not self.pretrained:
            self.blocks.apply(weight_init)

        if add_extra_block:
            # last block (keeps the same resolution of x16)
            feature_size = num_features[-1] * extra_block_expand
            extra_block = nn.Sequential()
            extra_block.add_module('pool6', nn.MaxPool2d(kernel_size=[3, 3], stride=1, padding=1))
            extra_block.add_module('conv6', nn.Conv2d(num_features[-1], feature_size, 3, stride=1, padding=6, dilation=6))
            extra_block.add_module('relu6', nn.ReLU())
            extra_block.add_module('conv7', nn.Conv2d(feature_size, feature_size, 1, stride=1, padding=0))
            extra_block.add_module('relu7', nn.ReLU())
            extra_block.apply(weight_init)
            self.blocks.append(extra_block)
            num_features.append(feature_size)


        self.out_cls = nn.ModuleList()
        self.out_link = nn.ModuleList()
        for out_features in num_features:
            self.out_cls.append(nn.Conv2d(out_features, 2, 1, stride=1, padding=0))
            self.out_link.append(nn.Conv2d(out_features, 16, 1, stride=1, padding=0))

        self.final_1 = nn.Conv2d(2, 2, 1, stride=1, padding=0)
        self.final_2 = nn.Conv2d(16, 16, 1, stride=1, padding=0)

        self.num_blocks = len(self.blocks)
        self.last_branch = 0 if self.version == '2s' else 1

    def forward(self, x):
        outputs_backbone = self.num_blocks * [None]
        for i_block, block in enumerate(self.blocks):
            x = block(x)
            outputs_backbone[i_block] = (self.out_cls[i_block](x), self.out_link[i_block](x))

        # final output - text/no-text classification & link
        out = [None, None]
        i_branch = self.num_blocks - 1

        if self.add_extra_block:
            # assuming the last 2 outputs have the same resolution
            for i_out in range(2):
                out[i_out] = outputs_backbone[i_branch][i_out] + outputs_backbone[i_branch - 1][i_out]
            i_branch -= 2
        else:
            out = [outputs_backbone[i_branch][0], outputs_backbone[i_branch][1]]
            i_branch -= 1

        for i_branch in range(i_branch, self.last_branch - 1, -1):
            for i_out in range(2):
                out[i_out] = outputs_backbone[i_branch][i_out] \
                             + nn.functional.upsample(out[i_out], scale_factor=2, mode="bilinear", align_corners=True)

        out[0] = self.final_1(out[0])
        out[1] = self.final_2(out[1])
        return out


##### OLD CODE - vgg16 only #####
class Net(nn.Module):
    def __init__(self, version='2s', dilation=True):
        super(Net, self).__init__()
        self.version = version
        self.dilation = dilation

        # TODO: modify padding
        self.conv1_1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.relu1_1 = nn.ReLU()
        self.conv1_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.relu1_2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2_1 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.relu2_1 = nn.ReLU()
        self.conv2_2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.relu2_2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.conv3_1 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.relu3_1 = nn.ReLU()
        self.conv3_2 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.relu3_2 = nn.ReLU()
        self.conv3_3 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.relu3_3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2)
        self.conv4_1 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.relu4_1 = nn.ReLU()
        self.conv4_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.relu4_2 = nn.ReLU()
        self.conv4_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.relu4_3 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(2)
        self.conv5_1 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.relu5_1 = nn.ReLU()
        self.conv5_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.relu5_2 = nn.ReLU()
        self.conv5_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.relu5_3 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(kernel_size=[3, 3], stride=1, padding=1)
        if self.dilation:
            self.conv6 = nn.Conv2d(512, 1024, 3, stride=1, padding=6, dilation=6)
        else:
            self.conv6 = nn.Conv2d(512, 1024, 3, stride=1, padding=1)
        self.relu6 = nn.ReLU()
        self.conv7 = nn.Conv2d(1024, 1024, 1, stride=1, padding=0)
        self.relu7 = nn.ReLU()

        self.out1_1 = nn.Conv2d(128, 2, 1, stride=1, padding=0)
        self.out1_2 = nn.Conv2d(128, 16, 1, stride=1, padding=0)
        self.out2_1 = nn.Conv2d(256, 2, 1, stride=1, padding=0)
        self.out2_2 = nn.Conv2d(256, 16, 1, stride=1, padding=0)
        self.out3_1 = nn.Conv2d(512, 2, 1, stride=1, padding=0)
        self.out3_2 = nn.Conv2d(512, 16, 1, stride=1, padding=0)
        self.out4_1 = nn.Conv2d(512, 2, 1, stride=1, padding=0)
        self.out4_2 = nn.Conv2d(512, 16, 1, stride=1, padding=0)
        self.out5_1 = nn.Conv2d(1024, 2, 1, stride=1, padding=0)
        self.out5_2 = nn.Conv2d(1024, 16, 1, stride=1, padding=0)

        self.final_1 = nn.Conv2d(2, 2, 1, stride=1, padding=0)
        self.final_2 = nn.Conv2d(16, 16, 1, stride=1, padding=0)

    def forward(self, x):
        # print("forward1")
        x = self.pool1(self.relu1_2(self.conv1_2(self.relu1_1(self.conv1_1(x)))))
        # print("forward11")
        x = self.relu2_2(self.conv2_2(self.relu2_1(self.conv2_1(x))))
        # print("forward12")
        l1_1x = self.out1_1(x)
        # print("forward13")
        l1_2x = self.out1_2(x)
        # print("forward14")
        x = self.relu3_3(self.conv3_3(self.relu3_2(self.conv3_2(self.relu3_1(self.conv3_1(self.pool2(x)))))))
        # print("forward15")
        l2_1x = self.out2_1(x)
        # print("forward16")
        l2_2x = self.out2_2(x)
        # print("forward17")

        x = self.relu4_3(self.conv4_3(self.relu4_2(self.conv4_2(self.relu4_1(self.conv4_1(self.pool3(x)))))))
        l3_1x = self.out3_1(x)
        l3_2x = self.out3_2(x)
        x = self.relu5_3(self.conv5_3(self.relu5_2(self.conv5_2(self.relu5_1(self.conv5_1(self.pool4(x)))))))
        l4_1x = self.out4_1(x)
        l4_2x = self.out4_2(x)
        x = self.relu7(self.conv7(self.relu6(self.conv6(self.pool5(x)))))
        l5_1x = self.out5_1(x)
        l5_2x = self.out5_2(x)
        # print("forward3")

        upsample1_1 = nn.functional.upsample(l5_1x + l4_1x, scale_factor=2, mode="bilinear", align_corners=True)
        upsample2_1 = nn.functional.upsample(upsample1_1 + l3_1x, scale_factor=2, mode="bilinear", align_corners=True)
        if self.version == "2s":
            upsample3_1 = nn.functional.upsample(upsample2_1 + l2_1x, scale_factor=2, mode="bilinear", align_corners=True)
            out_1 = upsample3_1 + l1_1x
        else:
            out_1 = upsample2_1 + l2_1x
        # out_1 = self.final_1(out_1)
        # print("forward4")

        upsample1_2 = nn.functional.upsample(l5_2x + l4_2x, scale_factor=2, mode="bilinear", align_corners=True)
        upsample2_2 = nn.functional.upsample(upsample1_2 + l3_2x, scale_factor=2, mode="bilinear", align_corners=True)
        if self.version == "2s":
            upsample3_2 = nn.functional.upsample(upsample2_2 + l2_2x, scale_factor=2, mode="bilinear", align_corners=True)
            out_2 = upsample3_2 + l1_2x
        else:
            out_2 = upsample2_2 + l2_2x
        # out_2 = self.final_2(out_2)
        # print("forward5")

        return [out_1, out_2]

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
