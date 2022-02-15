import logging
import torch, torchaudio
import torch.nn.functional as F
from torch import nn


logger = logging.getLogger(__name__)


class ResNetSE(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.num_filters = [32, 64, 128, 256]
        self.layers = [2, 2, 2, 2]
        self.inplanes = self.num_filters[0]
        self.encoder_type = 'ASP'
        self.n_mels = 64
        self.dim_feat = 512
        self.log_input = True

        self.conv1 = nn.Conv2d(1, self.num_filters[0] , kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(self.num_filters[0])
        
        self.layer1 = self._make_layer(BasicBlock, self.num_filters[0], self.layers[0])
        self.layer2 = self._make_layer(BasicBlock, self.num_filters[1], self.layers[1], stride=(2, 2))
        self.layer3 = self._make_layer(BasicBlock, self.num_filters[2], self.layers[2], stride=(2, 2))
        self.layer4 = self._make_layer(BasicBlock, self.num_filters[3], self.layers[3], stride=(2, 2))

        self.instancenorm = nn.InstanceNorm1d(self.n_mels)
        self.torchfb = torch.nn.Sequential(
                PreEmphasis(),
                torchaudio.transforms.MelSpectrogram(sample_rate=16000, 
                                                     n_fft=512, 
                                                     win_length=400, 
                                                     hop_length=160, 
                                                     window_fn=torch.hamming_window, 
                                                     n_mels=self.n_mels)
                )

        outmap_size = int(self.n_mels/8)

        self.attention = nn.Sequential(
            nn.Conv1d(self.num_filters[3] * outmap_size, 128, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, self.num_filters[3] * outmap_size, kernel_size=1),
            nn.Softmax(dim=2),
            )

        if self.encoder_type == "SAP":
            out_dim = self.num_filters[3] * outmap_size
        elif self.encoder_type == "ASP":
            out_dim = self.num_filters[3] * outmap_size * 2
        else:
            raise ValueError('Undefined encoder')

        self.fc = nn.Linear(out_dim, self.dim_feat)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def _init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        with torch.no_grad():
            x = self.torchfb(x) + 1e-6
            if self.log_input: 
                x = x.log()
            x = self.instancenorm(x).unsqueeze(1)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.reshape(x.size()[0],-1,x.size()[-1])

        w = self.attention(x)

        if self.encoder_type == "SAP":
            x = torch.sum(x * w, dim=2)
        elif self.encoder_type == "ASP":
            mu = torch.sum(x * w, dim=2)
            sg = torch.sqrt( ( torch.sum((x**2) * w, dim=2) - mu**2 ).clamp(min=1e-5) )
            x = torch.cat((mu,sg), 1)

        x = x.view(x.size()[0], -1)
        x = self.fc(x)

        return x
    

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplane, outplane, stride=1, downsample=None, reduction=8):
        super().__init__()
        self.conv1 = nn.Conv2d(inplane, outplane, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outplane)
        self.conv2 = nn.Conv2d(outplane, outplane, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outplane)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(outplane, reduction)
        self.downsample = downsample

    def forward(self,x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample:
            residual = self.downsample(residual)
        
        out += residual
        out = self.relu(out)
        return out


class SELayer(nn.Module):
    
    def __init__(self, channel, reduction):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel//reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )
    
    def forward(self,x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class PreEmphasis(nn.Module):

    def __init__(self, coef = 0.97):
        super().__init__()
        self.coef = coef
        # make kernel
        # In pytorch, the convolution operation uses cross-correlation. So, filter is flipped.
        self.register_buffer(
            'flipped_filter', torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, input):
        assert len(input.size()) == 2, 'The number of dimensions of input tensor must be 2!'
        # reflect padding to match lengths of in/out
        input = input.unsqueeze(1)
        input = F.pad(input, (1, 0), 'reflect')
        return F.conv1d(input, self.flipped_filter).squeeze(1)