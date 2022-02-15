import torch, os, math, logging
import torch.nn as nn
import torch.optim
import torch.utils.data
from torch.nn.init import normal, constant
from model.resnet import resnet18
from model.resse import ResNetSE


logger = logging.getLogger(__name__)


class BaselineLSTM(nn.Module):
    def __init__(self, args):
        super(BaselineLSTM, self).__init__()
        self.args = args
        self.img_feature_dim = 256  # the dimension of the CNN feature to represent each frame

        self.video_encoder = resnet18(pretrained=False)

        self.video_encoder.fc2 = nn.Linear(1000, self.img_feature_dim)

        self.lstm = nn.LSTM(self.img_feature_dim, self.img_feature_dim, bidirectional=True ,num_layers=2, batch_first=True)

        self.audio_encoder = ResNetSE()

        # The linear layer that maps the LSTM with the 3 outputs
        self.last_layer1 = nn.Linear(4 * self.img_feature_dim, 128)

        self.last_layer2 = nn.Linear(128, 2)

        for param in self.parameters():
            param.requires_grad = True
        
        self._init_parameters()
        
        self.load_checkpoint()

    def forward(self, video, audio):
        N, D, C, H, W = video.shape

        video_out = self.video_encoder(video.view(N*D, C, H, W))
        video_out = video_out.view(N, D, self.img_feature_dim)

        lstm_out, _ = self.lstm(video_out)
        lstm_out = lstm_out[:, -1, :]

        audio_out = self.audio_encoder(audio)

        output = self.last_layer1(torch.cat((lstm_out, audio_out), dim=1))

        output = self.last_layer2(output)

        return output

    def load_checkpoint(self):
        if self.args.checkpoint is not None:
            if os.path.exists(self.args.checkpoint):
                logger.info(f'loading checkpoint {self.args.checkpoint}')
                state = torch.load(self.args.checkpoint, map_location=f'cuda:{self.args.rank}')
                if 'module' in list(state["state_dict"].keys())[0]:
                    state_dict = { k[7:]: v for k, v in state["state_dict"].items() }
                else:
                    state_dict = state["state_dict"]
                self.load_state_dict(state_dict)
        
    def _init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class GazeLSTM(nn.Module):
    def __init__(self, args):
        super(GazeLSTM, self).__init__()
        self.args = args
        self.img_feature_dim = 256  # the dimension of the CNN feature to represent each frame

        self.video_encoder = resnet18(pretrained=True)

        self.video_encoder.fc2 = nn.Linear(1000, self.img_feature_dim)

        self.lstm = nn.LSTM(self.img_feature_dim, self.img_feature_dim,bidirectional=True,num_layers=2,batch_first=True)

        # The linear layer that maps the LSTM with the 3 outputs
        self.last_layer = nn.Linear(2*self.img_feature_dim, 3)
        self.load_checkpoint()

    def forward(self, video):

        video_out = self.video_encoder(video.view((-1, 3) + video.size()[-2:]))

        video_out = video_out.view(video.size(0),7,self.img_feature_dim)

        lstm_out, _ = self.lstm(video_out)
        lstm_out = lstm_out[:,3,:]
        output = self.last_layer(lstm_out).view(-1,3)


        angular_output = output[:,:2]
        angular_output[:,0:1] = math.pi*nn.Tanh()(angular_output[:,0:1])
        angular_output[:,1:2] = (math.pi/2)*nn.Tanh()(angular_output[:,1:2])

        var = math.pi*nn.Sigmoid()(output[:,2:3])
        var = var.view(-1,1).expand(var.size(0),2)

        return angular_output, var

    def load_checkpoint(self):
        if self.args.checkpoint is not None:
            if os.path.exists(self.args.checkpoint):
                logger.info(f'loading checkpoint {self.args.checkpoint}')
                map_loc = f'cuda:{self.args.rank}' if torch.cuda.is_available() else 'cpu'
                state = torch.load(self.args.checkpoint, map_location=map_loc)
                if 'module' in list(state["state_dict"].keys())[0]:
                    state_dict = { k[7:]: v for k, v in state["state_dict"].items() }
                else:
                    state_dict = state["state_dict"]
                self.load_state_dict(state_dict)


if __name__ == '__main__':
    video_test = torch.FloatTensor(4,21,224,224)
    model_test = BaselineLSTM()
    out = model_test.forward(video_test)
