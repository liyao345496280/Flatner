import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        """
        相当于一个Residual block
        :param n_inputs: int, 输入通道数1
        :param n_outputs: int, 输出通道数1
        :param kernel_size: int, 卷积核尺寸
        :param stride: int, 步长，一般为1
        :param dilation: int, 膨胀系数
        :param padding: int, 填充系数
        :param dropout: float, dropout比率
        """
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        # 经过conv1，输出的size其实是(Batch, input_channel, seq_len - kernel_size + padding)input = input.permute(0,2,1)
        self.chomp1 = Chomp1d(padding)#裁剪掉多出来的padding部分，维持输出时间步为seq_len
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()


    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class AttentionFusion(nn.Module):
    def __init__(self, channel_size, num_levels):
        super(AttentionFusion, self).__init__()
        self.attention_weights = nn.Parameter(torch.randn(num_levels, channel_size))

    def forward(self, features):
        # 应用softmax函数获取归一化的注意力权重
        # 注意力权重形状将为[num_levels, 1, num_channels, 1]
        attention_scores = torch.softmax(self.attention_weights, dim=0).unsqueeze(1).unsqueeze(-1)

        # 扩展attention_scores以匹配features的batch和seq_length维度
        # 假设features形状为[num_levels, batch_size, num_channels, seq_length]
        # 通过广播，attention_scores将自动扩展到匹配的形状
        weighted_features = torch.sum(features * attention_scores, dim=0)
        return weighted_features


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.ModuleList(layers)
        self.attention_fusion = AttentionFusion(out_channels, num_levels)

    def forward(self, x):
        A = []  # 用于存储每层输出的列表
        for layer in self.network:
            x = layer(x)
            A.append(x)

        # 将每层输出堆叠成一个新的维度，以便应用注意力机制
        stacked_features = torch.stack(A, dim=0)
        # 应用注意力机制融合特征
        attention_output = self.attention_fusion(stacked_features)

        return attention_output
