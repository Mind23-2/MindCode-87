#from torch import nn

#from ops.basic_ops import ConsensusModule, Identity
from transforms import *
#from torch.nn.init import normal, constant

from mindspore import nn
from mindspore.ops import Reshape, operations as P
from inception_model import InceptionV3


class BasicConv2dBN(nn.Cell):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, pad_mode='same', padding=0, has_bias=False):
        super(BasicConv2dBN, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride,
                              pad_mode=pad_mode, padding=padding, weight_init='he_normal', has_bias=has_bias)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class InceptionBNBlock(nn.Cell):
    def __init__(self, in_channel, out_channel, sub_channels = [[64], [96, 128], [16, 32], [32]], is_training=True, has_bias=False):
        super(InceptionBNBlock, self).__init__()
        self.block0 = BasicConv2dBN(in_channel, out_channel=sub_channels[0][0], kernel_size = (1, 1), stride=1)

        self.block1_0 = BasicConv2dBN(in_channel, out_channel=sub_channels[1][0], kernel_size = (1, 1), stride=1)
        self.block1_1 = BasicConv2dBN(sub_channels[1][0], out_channel=sub_channels[1][1], kernel_size = (3, 3), stride=1)

        self.block2_0 = BasicConv2dBN(in_channel, out_channel=sub_channels[2][0], kernel_size = (1, 1), stride=1)
        self.block2_1 = BasicConv2dBN(sub_channels[2][0], out_channel=sub_channels[2][1], kernel_size = (5, 5), stride=1)

        self.block3_0 = nn.MaxPool2d(kernel_size=(3, 3), stride=1, pad_mode='same')
        self.block3_1 = BasicConv2dBN(in_channel, out_channel=sub_channels[3][0], kernel_size = (3, 3), stride=1)

        self.concat = P.Concat(axis=1)

        out_channel_cacul = sub_channels[0][0] + sub_channels[1][1] + sub_channels[2][1] + sub_channels[3][0]
        if(out_channel != out_channel_cacul):
            raise ValueError('[Inception Block] The sum of all sub output channels is not `' + str(out_channel) + '`, but it is `' + str(out_channel_cacul) + '` now. ')
        pass

    def construct(self, x):
        y0 = self.block0(x)

        y1 = self.block1_0(x)
        y1 = self.block1_1(y1)

        y2 = self.block2_0(x)
        y2 = self.block2_1(y2)

        y3 = self.block3_0(x)
        y3 = self.block3_1(y3)

        y = self.concat((y0, y1, y2, y3))
        return y

class InceptionBN(nn.Cell):
    def __init__(self, in_channel, num_class, is_training=True, has_bias=False, dropout_keep_prob=0.8, include_top=True):
        super(InceptionBN, self).__init__()

        self.layer1 = BasicConv2dBN(in_channel, out_channel=64, kernel_size = (7, 7), stride=2)

        self.layer2a = nn.MaxPool2d(kernel_size=(3, 3), stride=2, pad_mode='same')
        self.layer2b = BasicConv2dBN(in_channel=64, out_channel=64, kernel_size = (1, 1), stride=1)
        self.layer2c = BasicConv2dBN(in_channel=64, out_channel=192, kernel_size = (3, 3), stride=1)

        self.layer2 = nn.MaxPool2d(kernel_size=(3, 3), stride=2, pad_mode='same')

        self.layer3a = InceptionBNBlock(192, 256, sub_channels=[[64], [96, 128], [16, 32], [32]])
        self.layer3b = InceptionBNBlock(256, 480, sub_channels=[[128], [128, 192], [32, 96], [64]])

        self.layer3 = nn.MaxPool2d(kernel_size=(3, 3), stride=2, pad_mode='same')

        self.layer4a = InceptionBNBlock(480, 512, sub_channels=[[192], [96, 208], [16, 48], [64]])
        self.layer4b = InceptionBNBlock(512, 512, sub_channels=[[160], [112, 224], [24, 64], [64]])
        self.layer4c = InceptionBNBlock(512, 512, sub_channels=[[128], [128, 256], [24, 64], [64]])
        self.layer4d = InceptionBNBlock(512, 528, sub_channels=[[112], [144, 288], [32, 64], [64]])
        self.layer4e = InceptionBNBlock(528, 832, sub_channels=[[256], [160, 320], [32, 128], [128]])

        self.layer4 = nn.MaxPool2d(kernel_size=(3, 3), stride=2, pad_mode='same')

        self.layer5a = InceptionBNBlock(832, 832, sub_channels=[[256], [160, 320], [32, 128], [128]])
        self.layer5b = InceptionBNBlock(832, 1024, sub_channels=[[384], [192, 384], [48, 128], [128]])

        self.layer5 = nn.AvgPool2d(kernel_size=(7, 7), stride=1, pad_mode='valid', data_format='NCHW')
        self.reshape = Reshape()
        self.layer5_drop = nn.Dropout(keep_prob=dropout_keep_prob)
        self.layer6 = nn.Dense(1024, 1000)
        self.layer_out = nn.Dense(1000, num_class)

    def construct(self, x):
        x = self.layer1(x)

        x = self.layer2a(x)
        x = self.layer2b(x)
        x = self.layer2c(x)
        x = self.layer2(x)

        x = self.layer3a(x)
        x = self.layer3b(x)
        x = self.layer3(x)

        x = self.layer4a(x)
        x = self.layer4b(x)
        x = self.layer4c(x)
        x = self.layer4d(x)
        x = self.layer4e(x)
        x = self.layer4(x)

        x = self.layer5a(x)
        x = self.layer5b(x)
        x = self.layer5(x)
        x = self.reshape(x, (x.shape[0], 1024))
        x = self.layer5_drop(x)

        x = self.layer6(x)
        x = self.layer_out(x)
        return x

class TSN(nn.Cell):
    def __init__(self, in_channel = 3, num_classes = 101):
        super(TSN, self).__init__()
        self.base_model_1 = InceptionBN(in_channel, num_classes)
        self.base_model_2 = InceptionBN(in_channel, num_classes)
        self.base_model_3 = InceptionBN(in_channel, num_classes)
        pass
    
    def construct(self, x):
        x1 = self.base_model_1(x)
        x2 = self.base_model_2(x)
        x3 = self.base_model_3(x)
        return x1 + x2 + x3
        pass




















class TSN_light(nn.Cell):
    def __init__(self, num_classes = 101):
        super(TSN_light, self).__init__()
        self.base_model_1 = InceptionV3(num_classes)
        self.base_model_2 = InceptionV3(num_classes)
        self.base_model_3 = InceptionV3(num_classes)
        pass
    
    def construct(self, x):
        x1 = self.base_model_1(x)
        x2 = self.base_model_2(x)
        x3 = self.base_model_3(x)
        return x1 + x2 + x3
        pass


class Inception(nn.Cell):
    def __init__(self, in_channels = 3, sub_channels = [[64], [48, 64], [64, 96, 96], [32]]):
        super(Inception, self).__init__()
        self.net0 = nn.Conv2d(in_channels = in_channels, out_channels = sub_channels[0][0], kernel_size = (1, 1), stride=1, pad_mode='same', padding=0, dilation=1, group=1, has_bias=True, weight_init='normal', bias_init='zeros', data_format='NCHW')
        self.net0_bn = nn.BatchNorm2d(num_features = sub_channels[0][0], eps=1e-05, momentum=0.9, affine=True, gamma_init='ones', beta_init='zeros', moving_mean_init='zeros', moving_var_init='ones', use_batch_statistics=None, data_format='NCHW')

        self.net1_1 = nn.Conv2d(in_channels = in_channels, out_channels = sub_channels[1][0], kernel_size = (1, 1), stride=1, pad_mode='same', padding=0, dilation=1, group=1, has_bias=True, weight_init='normal', bias_init='zeros', data_format='NCHW')
        self.net1_2 = nn.Conv2d(in_channels = sub_channels[1][0], out_channels = sub_channels[1][1], kernel_size = (5, 5), stride=1, pad_mode='same', padding=0, dilation=1, group=1, has_bias=True, weight_init='normal', bias_init='zeros', data_format='NCHW')
        self.net1_bn = nn.BatchNorm2d(num_features = sub_channels[1][1], eps=1e-05, momentum=0.9, affine=True, gamma_init='ones', beta_init='zeros', moving_mean_init='zeros', moving_var_init='ones', use_batch_statistics=None, data_format='NCHW')

        self.net2_1 = nn.Conv2d(in_channels = in_channels, out_channels = sub_channels[2][0], kernel_size = (1, 1), stride=1, pad_mode='same', padding=0, dilation=1, group=1, has_bias=True, weight_init='normal', bias_init='zeros', data_format='NCHW')
        self.net2_2 = nn.Conv2d(in_channels = sub_channels[2][0], out_channels = sub_channels[2][1], kernel_size = (3, 3), stride=1, pad_mode='same', padding=0, dilation=1, group=1, has_bias=True, weight_init='normal', bias_init='zeros', data_format='NCHW')
        self.net2_3 = nn.Conv2d(in_channels = sub_channels[2][1], out_channels = sub_channels[2][2], kernel_size = (3, 3), stride=1, pad_mode='same', padding=0, dilation=1, group=1, has_bias=True, weight_init='normal', bias_init='zeros', data_format='NCHW')
        self.net2_bn = nn.BatchNorm2d(num_features = sub_channels[2][2], eps=1e-05, momentum=0.9, affine=True, gamma_init='ones', beta_init='zeros', moving_mean_init='zeros', moving_var_init='ones', use_batch_statistics=None, data_format='NCHW')

        self.net3_pool = nn.AvgPool2d(kernel_size=(3, 3), stride=1, pad_mode='same', data_format='NCHW')
        self.net3_1 = nn.Conv2d(in_channels = in_channels, out_channels = sub_channels[3][0], kernel_size = (1, 1), stride=1, pad_mode='same', padding=0, dilation=1, group=1, has_bias=True, weight_init='normal', bias_init='zeros', data_format='NCHW')
        self.net3_bn = nn.BatchNorm2d(num_features = sub_channels[3][0], eps=1e-05, momentum=0.9, affine=True, gamma_init='ones', beta_init='zeros', moving_mean_init='zeros', moving_var_init='ones', use_batch_statistics=None, data_format='NCHW')

        self.concat = P.Concat(axis=1)

    def construct(self, x):
        x0 = self.net0(x)
        x0 = self.net0_bn(x0)

        x1 = self.net1_1(x)
        x1 = self.net1_2(x1)
        x1 = self.net1_bn(x1)

        x2 = self.net2_1(x)
        x2 = self.net2_2(x2)
        x2 = self.net2_3(x2)
        x2 = self.net2_bn(x2)

        x3 = self.net3_pool(x)
        x3 = self.net3_1(x3)
        x3 = self.net3_bn(x3)

        y = self.concat((x0, x1, x2, x3))
        pass

class InceptionV2(nn.Cell):
    def __init__(self):
        super(InceptionV2, self).__init__()
        self.net1 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = (7, 7), stride=2, pad_mode='same', padding=0, dilation=1, group=1, has_bias=True, weight_init='normal', bias_init='zeros', data_format='NCHW')
        self.net1_bn = nn.BatchNorm2d(num_features = 32, eps=1e-05, momentum=0.9, affine=True, gamma_init='ones', beta_init='zeros', moving_mean_init='zeros', moving_var_init='ones', use_batch_statistics=None, data_format='NCHW')
        
        self.net2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = (1, 1), stride=1, pad_mode='same', padding=0, dilation=1, group=1, has_bias=True, weight_init='normal', bias_init='zeros', data_format='NCHW')
        self.net2_bn = nn.BatchNorm2d(num_features = 32, eps=1e-05, momentum=0.9, affine=True, gamma_init='ones', beta_init='zeros', moving_mean_init='zeros', moving_var_init='ones', use_batch_statistics=None, data_format='NCHW')
        
        self.net3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = (3, 3), stride=1, pad_mode='same', padding=0, dilation=1, group=1, has_bias=True, weight_init='normal', bias_init='zeros', data_format='NCHW')
        self.net3_bn = nn.BatchNorm2d(num_features = 64, eps=1e-05, momentum=0.9, affine=True, gamma_init='ones', beta_init='zeros', moving_mean_init='zeros', moving_var_init='ones', use_batch_statistics=None, data_format='NCHW')
        
        self.net3_pool = nn.MaxPool2d(kernel_size=3, stride=2)

        self.net4 = nn.Conv2d(in_channels = 64, out_channels = 80, kernel_size = (7, 7), stride=2, pad_mode='same', padding=0, dilation=1, group=1, has_bias=True, weight_init='normal', bias_init='zeros', data_format='NCHW')
        self.net4_bn = nn.BatchNorm2d(num_features = 80, eps=1e-05, momentum=0.9, affine=True, gamma_init='ones', beta_init='zeros', moving_mean_init='zeros', moving_var_init='ones', use_batch_statistics=None, data_format='NCHW')
        
        self.net5 = nn.Conv2d(in_channels = 80, out_channels = 192, kernel_size = (1, 1), stride=1, pad_mode='same', padding=0, dilation=1, group=1, has_bias=True, weight_init='normal', bias_init='zeros', data_format='NCHW')
        self.net5_bn = nn.BatchNorm2d(num_features = 192, eps=1e-05, momentum=0.9, affine=True, gamma_init='ones', beta_init='zeros', moving_mean_init='zeros', moving_var_init='ones', use_batch_statistics=None, data_format='NCHW')
        
        self.net6 = nn.Conv2d(in_channels = 192, out_channels = 288, kernel_size = (3, 3), stride=1, pad_mode='same', padding=0, dilation=1, group=1, has_bias=True, weight_init='normal', bias_init='zeros', data_format='NCHW')
        self.net6_bn = nn.BatchNorm2d(num_features = 288, eps=1e-05, momentum=0.9, affine=True, gamma_init='ones', beta_init='zeros', moving_mean_init='zeros', moving_var_init='ones', use_batch_statistics=None, data_format='NCHW')

        self.relu = nn.ReLU()

        self.inception_1 = Inception(in_channels = 288, sub_channels = [[64], [48, 64], [64, 96, 96], [32]])
        self.inception_2 = Inception(in_channels = 768, sub_channels = [[64], [48, 64], [64, 96, 96], [64]])
        self.inception_3 = Inception(in_channels = 1280, sub_channels = [[64], [48, 64], [64, 96, 96], [64]])

        self.net7_pool = nn.MaxPool2d(kernel_size=8, stride=1)


    def construct(self, x):
        pass

    pass

class TSN_pytorch(nn.Cell):
    def __init__(self, num_class, num_segments, modality, base_model='resnet101', new_length=None,
                 consensus_type='avg', before_softmax=True, dropout=0.8, crop_num=1, partial_bn=True):
        super(TSN, self).__init__()
        self.modality = modality
        self.num_segments = num_segments
        self.reshape = True
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.crop_num = crop_num
        self.consensus_type = consensus_type
        if not before_softmax and consensus_type != 'avg':
            raise ValueError("Only avg consensus can be used after Softmax")

        if new_length is None:
            self.new_length = 1 if modality == "RGB" else 5
        else:
            self.new_length = new_length

        print(("""
Initializing TSN with base model: {}.
TSN Configurations:
    input_modality:     {}
    num_segments:       {}
    new_length:         {}
    consensus_module:   {}
    dropout_ratio:      {}
        """.format(base_model, self.modality, self.num_segments, self.new_length, consensus_type, self.dropout)))

        self._prepare_base_model(base_model)

        feature_dim = self._prepare_tsn(num_class)

        if self.modality == 'Flow':
            print("Converting the ImageNet model to a flow init model")
            self.base_model = self._construct_flow_model(self.base_model)
            print("Done. Flow model ready...")
        elif self.modality == 'RGBDiff':
            print("Converting the ImageNet model to RGB+Diff init model")
            self.base_model = self._construct_diff_model(self.base_model)
            print("Done. RGBDiff model ready.")

        self.consensus = ConsensusModule(consensus_type)

        if not self.before_softmax:
            self.softmax = nn.Softmax()

        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)

    def _prepare_tsn(self, num_class):
        feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
        if self.dropout == 0:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Linear(feature_dim, num_class))
            self.new_fc = None
        else:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))
            self.new_fc = nn.Linear(feature_dim, num_class)

        std = 0.001
        if self.new_fc is None:
            normal(getattr(self.base_model, self.base_model.last_layer_name).weight, 0, std)
            constant(getattr(self.base_model, self.base_model.last_layer_name).bias, 0)
        else:
            normal(self.new_fc.weight, 0, std)
            constant(self.new_fc.bias, 0)
        return feature_dim

    def _prepare_base_model(self, base_model):

        if 'resnet' in base_model or 'vgg' in base_model:
            self.base_model = getattr(torchvision.models, base_model)(True)
            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]

            if self.modality == 'Flow':
                self.input_mean = [0.5]
                self.input_std = [np.mean(self.input_std)]
            elif self.modality == 'RGBDiff':
                self.input_mean = [0.485, 0.456, 0.406] + [0] * 3 * self.new_length
                self.input_std = self.input_std + [np.mean(self.input_std) * 2] * 3 * self.new_length
        elif base_model == 'BNInception':
            import tf_model_zoo
            self.base_model = getattr(tf_model_zoo, base_model)()
            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [104, 117, 128]
            self.input_std = [1]

            if self.modality == 'Flow':
                self.input_mean = [128]
            elif self.modality == 'RGBDiff':
                self.input_mean = self.input_mean * (1 + self.new_length)

        elif 'inception' in base_model:
            import tf_model_zoo
            self.base_model = getattr(tf_model_zoo, base_model)()
            self.base_model.last_layer_name = 'classif'
            self.input_size = 299
            self.input_mean = [0.5]
            self.input_std = [0.5]
        else:
            raise ValueError('Unknown base model: {}'.format(base_model))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(TSN, self).train(mode)
        count = 0
        if self._enable_pbn:
            print("Freezing BatchNorm2D except the first one.")
            for m in self.base_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()

                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False

    def partialBN(self, enable):
        self._enable_pbn = enable

    def get_optim_policies(self):
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        bn = []

        conv_cnt = 0
        bn_cnt = 0
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d):
                ps = list(m.parameters())
                conv_cnt += 1
                if conv_cnt == 1:
                    first_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        first_conv_bias.append(ps[1])
                else:
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                normal_weight.append(ps[0])
                if len(ps) == 2:
                    normal_bias.append(ps[1])
                  
            elif isinstance(m, torch.nn.BatchNorm1d):
                bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.BatchNorm2d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

        return [
            {'params': first_conv_weight, 'lr_mult': 5 if self.modality == 'Flow' else 1, 'decay_mult': 1,
             'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 10 if self.modality == 'Flow' else 2, 'decay_mult': 0,
             'name': "first_conv_bias"},
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"},
        ]

    def construct(self, input):
        sample_len = (3 if self.modality == "RGB" else 2) * self.new_length

        if self.modality == 'RGBDiff':
            sample_len = 3 * self.new_length
            input = self._get_diff(input)

        base_out = self.base_model(input.view((-1, sample_len) + input.size()[-2:]))

        if self.dropout > 0:
            base_out = self.new_fc(base_out)

        if not self.before_softmax:
            base_out = self.softmax(base_out)
        if self.reshape:
            base_out = base_out.view((-1, self.num_segments) + base_out.size()[1:])

        output = self.consensus(base_out)
        return output.squeeze(1)

    def _get_diff(self, input, keep_rgb=False):
        input_c = 3 if self.modality in ["RGB", "RGBDiff"] else 2
        input_view = input.view((-1, self.num_segments, self.new_length + 1, input_c,) + input.size()[2:])
        if keep_rgb:
            new_data = input_view.clone()
        else:
            new_data = input_view[:, :, 1:, :, :, :].clone()

        for x in reversed(list(range(1, self.new_length + 1))):
            if keep_rgb:
                new_data[:, :, x, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]
            else:
                new_data[:, :, x - 1, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]

        return new_data


    def _construct_flow_model(self, base_model):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (2 * self.new_length, ) + kernel_size[2:]
        new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()

        new_conv = nn.Conv2d(2 * self.new_length, conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7] # remove .weight suffix to get the layer name

        # replace the first convlution layer
        setattr(container, layer_name, new_conv)
        return base_model

    def _construct_diff_model(self, base_model, keep_rgb=False):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        if not keep_rgb:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()
        else:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernels = torch.cat((params[0].data, params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()),
                                    1)
            new_kernel_size = kernel_size[:1] + (3 + 3 * self.new_length,) + kernel_size[2:]

        new_conv = nn.Conv2d(new_kernel_size[1], conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data  # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7]  # remove .weight suffix to get the layer name

        # replace the first convolution layer
        setattr(container, layer_name, new_conv)
        return base_model

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224

    def get_augmentation(self):
        if self.modality == 'RGB':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
                                                   GroupRandomHorizontalFlip(is_flow=False)])
        elif self.modality == 'Flow':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                                                   GroupRandomHorizontalFlip(is_flow=True)])
        elif self.modality == 'RGBDiff':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                                                   GroupRandomHorizontalFlip(is_flow=False)])


