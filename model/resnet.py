import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU()
        if stride !=1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False), 
                                          nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + self.shortcut(identity))
        return out


class BottleNeck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding= 0, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3   = nn.BatchNorm2d(out_channels*self.expansion)
        self.relu  = nn.ReLU()
        if stride !=1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels*self.expansion, kernel_size=1, stride=stride, bias=False), 
                                          nn.BatchNorm2d(out_channels * self.expansion))
        else:
            self.shortcut = nn.Identity()
            
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.relu(out + self.shortcut(identity))
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):  
        super().__init__()
        self.in_channels = 64
        
        self.init_conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.init_bn = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU()
        
        self.layer2 = self._make_layer(block, 64,  layers[0], stride=1)
        self.layer3 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer4 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer5 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(in_features=512*block.expansion,out_features=num_classes)
        
    def _make_layer(self,block, planes, num_layer, stride):
        layers = []
        layers.append(block(self.in_channels, planes, stride = stride))
        
        self.in_channels = planes * block.expansion
        
        for _ in range(1,num_layer):
            layers.append(block(self.in_channels, planes))  
                        
        return nn.Sequential(*layers)
    
    def forward(self,x):
        out = self.relu(self.init_bn(self.init_conv(x)))
        out = self.maxpool(out)
        
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        
        out = self.fc(torch.flatten(self.avgpool(out),1))
        
        return out

class ResNet_CIFAR100(nn.Module):
    def __init__(self, block, layers, num_classes=1000):  
        super().__init__()
        self.in_channels = 64
        
        self.init_conv = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.init_bn = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU()
        
        self.layer2 = self._make_layer(block, 64,  layers[0], stride=1)
        self.layer3 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer4 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer5 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(in_features=512*block.expansion,out_features=num_classes)
        
    def _make_layer(self,block, planes, num_layer, stride):
        layers = []
        layers.append(block(self.in_channels, planes, stride = stride))
        
        self.in_channels = planes * block.expansion
        
        for _ in range(1,num_layer):
            layers.append(block(self.in_channels, planes))  
                        
        return nn.Sequential(*layers)
    
    def forward(self,x):
        out = self.relu(self.init_bn(self.init_conv(x)))
        #out = self.maxpool(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        
        out = self.fc(torch.flatten(self.avgpool(out),1))
        
        return out


RESNET_CONFIG = {
    "resnet18"  : (BasicBlock, [2, 2, 2,  2]),
    "resnet34"  : (BasicBlock, [3, 4, 6,  3]),
    "resnet50"  : (BottleNeck, [3, 4, 6,  3]),
    "resnet101" : (BottleNeck, [3, 4, 23, 3]),
    "resnet152" : (BottleNeck, [3, 8, 36, 3])
}
def get_resnet(arch, num_classes=1000):
    if arch not in RESNET_CONFIG:
        raise ValueError(
            f"Unknown architecture: {arch}."
            f"Available: {list(RESNET_CONFIG.keys())}"
            )
    
    block, layers = RESNET_CONFIG[arch]
    return ResNet(block, layers, num_classes)

def get_resnet_cifar100(arch, num_classes=1000):
    if arch not in RESNET_CONFIG:
        raise ValueError(
            f"Unknown architecture: {arch}."
            f"Available: {list(RESNET_CONFIG.keys())}"
            )
    
    block, layers = RESNET_CONFIG[arch]
    return ResNet_CIFAR100(block, layers, num_classes)



if __name__ == "__main__":
    
    #BasicBlock
    # input1 = torch.randn(2,64,56,56)
    # model1 = BasicBlock(in_channels=64, out_channels=64)
    # print(model1(input1).shape) #[2,64,56,56]
    
    input = torch.randn(2,3,224,224)
    arch = 'resnet50'
    num_classes = 1000
    
    model = get_resnet(arch, num_classes)
    out = model(input)
    
    print(model)
    print(out.shape) #[2, 1000]



# print(images.size()) # torch.Size([128, 3, 32, 32]) == print(images.shape)        
# [B,C,H,W] = [Batch size, Channel, Height, Width]
   
# Resnet 구조
# Conv1 7x7,64 stride 2 -> 112x112
# maxpool 3x3 maxpool, stride 2 
# Conv2  ->56x56
# Conv3  ->28x28
# Conv4  ->14x14
# Conv5  ->7x7
# FC average pool, 1000-d fc
# out = [batch_size, num_classes]

# 공식처럼 기억하기
# class MyModel(nn.Module):
#   def __init__(self):
#       super().__init__()
#   def forward(self,x):
#       return out

# nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
# shortcut = conv(in_channels, out_channels, kernel_size=1, stride=stride)
