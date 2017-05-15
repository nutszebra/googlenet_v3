import numpy as np
import functools
import chainer.links as L
import chainer.functions as F
from collections import defaultdict
import nutszebra_chainer


class Conv_BN_ReLU(nutszebra_chainer.Model):

    def __init__(self, in_channel, out_channel, filter_size=(3, 3), stride=(1, 1), pad=(1, 1)):
        super(Conv_BN_ReLU, self).__init__(
            conv=L.Convolution2D(in_channel, out_channel, filter_size, stride, pad),
            bn=L.BatchNormalization(out_channel),
        )

    def weight_initialization(self):
        self.conv.W.data = self.weight_relu_initialization(self.conv)
        self.conv.b.data = self.bias_initialization(self.conv, constant=0)

    def __call__(self, x, train=False):
        return F.relu(self.bn(self.conv(x), test=not train))

    def count_parameters(self):
        return functools.reduce(lambda a, b: a * b, self.conv.W.data.shape)


class Inception_A(nutszebra_chainer.Model):

    def __init__(self, in_channel, conv5x5=(64, 96, 96), conv3x3=(48, 64), pool1x1=32, conv1x1=64, pool='ave', stride=1):
        super(Inception_A, self).__init__()
        modules = []
        modules.append(('conv5x5_1', Conv_BN_ReLU(in_channel, conv5x5[0], 1, 1, 0)))
        modules.append(('conv5x5_2', Conv_BN_ReLU(conv5x5[0], conv5x5[1], 3, 1, 1)))
        modules.append(('conv5x5_3', Conv_BN_ReLU(conv5x5[1], conv5x5[2], 3, stride, 1)))
        modules.append(('conv3x3_1', Conv_BN_ReLU(in_channel, conv3x3[0], 1, 1, 0)))
        modules.append(('conv3x3_2', Conv_BN_ReLU(conv3x3[0], conv3x3[1], 3, stride, 1)))
        if stride == 1:
            modules.append(('conv_pool', Conv_BN_ReLU(in_channel, pool1x1, 1, 1, 0)))
            modules.append(('conv1x1', Conv_BN_ReLU(in_channel, conv1x1, 1, 1, 0)))
        # register layers
        [self.add_link(*link) for link in modules]
        self.modules = modules
        self.pool = pool
        self.stride = stride

    def weight_initialization(self):
        for name, link in self.modules:
            link.weight_initialization()

    @staticmethod
    def max_or_ave(word='ave'):
        if word == 'ave':
            return F.average_pooling_2d
        return F.max_pooling_2d

    def __call__(self, x, train=False):
        pool = Inception_A.max_or_ave(self.pool)
        a = self.conv5x5_1(x, train)
        a = self.conv5x5_2(a, train)
        a = self.conv5x5_3(a, train)
        b = self.conv3x3_1(x, train)
        b = self.conv3x3_2(b, train)
        c = pool(x, ksize=3, stride=self.stride, pad=1)
        if not self.stride == 1:
            return F.concat((a, b, c), axis=1)
        c = self.conv_pool(c, train)
        d = self.conv1x1(x, train)
        return F.concat((a, b, c, d), axis=1)

    def count_parameters(self):
        count = 0
        for name, link in self.modules:
            count += link.count_parameters()
        return count


class Inception_B(nutszebra_chainer.Model):

    def __init__(self, in_channel, double_convnxn=(128, 128, 128, 128, 192), convnxn=(128, 128, 192), pool1x1=192, conv1x1=192, pool='ave', stride=1, n=7):
        super(Inception_B, self).__init__()
        modules = []
        if stride == 1:
            modules.append(('double_convnxn_1', Conv_BN_ReLU(in_channel, double_convnxn[0], 1, 1, 0)))
            modules.append(('double_convnxn_2', Conv_BN_ReLU(double_convnxn[0], double_convnxn[1], (1, n), 1, (0, int(n / 2)))))
            modules.append(('double_convnxn_3', Conv_BN_ReLU(double_convnxn[1], double_convnxn[2], (n, 1), 1, (int(n / 2), 0))))
            modules.append(('double_convnxn_4', Conv_BN_ReLU(double_convnxn[2], double_convnxn[3], (1, n), 1, (0, int(n / 2)))))
            modules.append(('double_convnxn_5', Conv_BN_ReLU(double_convnxn[3], double_convnxn[4], (n, 1), 1, (int(n / 2), 0))))
            modules.append(('convnxn_1', Conv_BN_ReLU(in_channel, convnxn[0], 1, 1, 0)))
            modules.append(('convnxn_2', Conv_BN_ReLU(convnxn[0], convnxn[1], (1, n), 1, (0, int(n / 2)))))
            modules.append(('convnxn_3', Conv_BN_ReLU(convnxn[1], convnxn[2], (n, 1), 1, (int(n / 2), 0))))
            modules.append(('conv_pool', Conv_BN_ReLU(in_channel, pool1x1, 1, 1, 0)))
            modules.append(('conv1x1', Conv_BN_ReLU(in_channel, conv1x1, 1, 1, 0)))
        else:
            modules.append(('double_convnxn_1', Conv_BN_ReLU(in_channel, double_convnxn[0], 1, 1, 0)))
            modules.append(('double_convnxn_2', Conv_BN_ReLU(double_convnxn[0], double_convnxn[1], (1, n), 1, (0, int(n / 2)))))
            modules.append(('double_convnxn_3', Conv_BN_ReLU(double_convnxn[1], double_convnxn[2], (n, 1), 1, (int(n / 2), 0))))
            modules.append(('double_convnxn_4', Conv_BN_ReLU(double_convnxn[2], double_convnxn[3], 3, stride, 1)))
            modules.append(('convnxn_1', Conv_BN_ReLU(in_channel, convnxn[0], 1, 1, 0)))
            modules.append(('convnxn_2', Conv_BN_ReLU(convnxn[0], convnxn[1], 3, stride, 1)))
        # register layers
        [self.add_link(*link) for link in modules]
        self.modules = modules
        self.pool = pool
        self.stride = stride
        self.n = n

    def weight_initialization(self):
        for name, link in self.modules:
            link.weight_initialization()

    @staticmethod
    def max_or_ave(word='ave'):
        if word == 'ave':
            return F.average_pooling_2d
        return F.max_pooling_2d

    def __call__(self, x, train=False):
        pool = Inception_A.max_or_ave(self.pool)
        if self.stride == 1:
            a = self.double_convnxn_1(x, train)
            a = self.double_convnxn_2(a, train)
            a = self.double_convnxn_3(a, train)
            a = self.double_convnxn_4(a, train)
            a = self.double_convnxn_5(a, train)
            b = self.convnxn_1(x, train)
            b = self.convnxn_2(b, train)
            b = self.convnxn_3(b, train)
            c = pool(x, ksize=3, stride=self.stride, pad=1)
            c = self.conv_pool(c, train)
            d = self.conv1x1(x, train)
            return F.concat((a, b, c, d), axis=1)
        else:
            a = self.double_convnxn_1(x, train)
            a = self.double_convnxn_2(a, train)
            a = self.double_convnxn_3(a, train)
            a = self.double_convnxn_4(a, train)
            b = self.convnxn_1(x, train)
            b = self.convnxn_2(b, train)
            c = pool(x, ksize=3, stride=self.stride, pad=1)
            return F.concat((a, b, c), axis=1)

    def count_parameters(self):
        count = 0
        for name, link in self.modules:
            count += link.count_parameters()
        return count


class Inception_C(nutszebra_chainer.Model):

    def __init__(self, in_channel, conv3x3U=(448, 384, 384, 384), conv1x1U=(384, 384, 384), pool1x1=192, conv1x1=320, pool='ave', n=3):
        super(Inception_C, self).__init__()
        modules = []
        modules.append(('conv3x3U_1', Conv_BN_ReLU(in_channel, conv3x3U[0], 1, 1, 0)))
        modules.append(('conv3x3U_2', Conv_BN_ReLU(conv3x3U[0], conv3x3U[1], 3, 1, 1)))
        modules.append(('conv3x3U_3', Conv_BN_ReLU(conv3x3U[1], conv3x3U[2], (1, n), 1, (0, int(n / 2)))))
        modules.append(('conv3x3U_4', Conv_BN_ReLU(conv3x3U[1], conv3x3U[3], (n, 1), 1, (int(n / 2), 0))))
        modules.append(('conv1x1U_1', Conv_BN_ReLU(in_channel, conv1x1U[0], 1, 1, 0)))
        modules.append(('conv1x1U_2', Conv_BN_ReLU(conv1x1U[0], conv1x1U[1], (1, n), 1, (0, int(n / 2)))))
        modules.append(('conv1x1U_3', Conv_BN_ReLU(conv1x1U[0], conv1x1U[2], (n, 1), 1, (int(n / 2), 0))))
        modules.append(('conv_pool', Conv_BN_ReLU(in_channel, pool1x1, 1, 1, 0)))
        modules.append(('conv1x1', Conv_BN_ReLU(in_channel, conv1x1, 1, 1, 0)))
        # register layers
        [self.add_link(*link) for link in modules]
        self.modules = modules
        self.pool = pool
        self.n = n

    def weight_initialization(self):
        for name, link in self.modules:
            link.weight_initialization()

    @staticmethod
    def max_or_ave(word='ave'):
        if word == 'ave':
            return F.average_pooling_2d
        return F.max_pooling_2d

    def __call__(self, x, train=False):
        pool = Inception_A.max_or_ave(self.pool)
        a = self.conv3x3U_1(x, train)
        a = self.conv3x3U_2(a, train)
        a_1 = self.conv3x3U_3(a, train)
        a_2 = self.conv3x3U_4(a, train)
        b = self.conv1x1U_1(x, train)
        b_1 = self.conv1x1U_2(b, train)
        b_2 = self.conv1x1U_3(b, train)
        c = pool(x, ksize=3, stride=1, pad=1)
        c = self.conv_pool(c, train)
        d = self.conv1x1(x, train)
        return F.concat((a_1, a_2, b_1, b_2, c, d), axis=1)

    def count_parameters(self):
        count = 0
        for name, link in self.modules:
            count += link.count_parameters()
        return count


class Googlenet(nutszebra_chainer.Model):

    def __init__(self, category_num):
        super(Googlenet, self).__init__()
        modules = []
        modules += [('conv1', Conv_BN_ReLU(3, 32, 3, 2, 1))]
        modules += [('conv2', Conv_BN_ReLU(32, 32, 3, 1, 0))]
        modules += [('conv3', Conv_BN_ReLU(32, 64, 3, 1, 1))]
        modules += [('conv4', Conv_BN_ReLU(64, 64, 3, 1, 0))]
        modules += [('conv5', Conv_BN_ReLU(64, 80, 3, 2, 1))]
        modules += [('conv6', Conv_BN_ReLU(80, 192, 3, 1, 0))]
        modules += [('inception_f5_1', Inception_A(192, (64, 96, 96), (48, 64), 32, 64, 'ave', 1))]
        modules += [('inception_f5_2', Inception_A(256, (64, 96, 96), (48, 64), 64, 64, 'ave', 1))]
        modules += [('inception_f5_3', Inception_A(288, (64, 96, 96), (288, 384), 0, 0, 'max', 2))]
        modules += [('inception_f6_1', Inception_B(768, (128, 128, 128, 128, 192), (128, 128, 192), 192, 192, 'ave', 1, 7))]
        modules += [('inception_f6_2', Inception_B(768, (160, 160, 160, 160, 192), (160, 160, 192), 192, 192, 'ave', 1, 7))]
        modules += [('inception_f6_3', Inception_B(768, (160, 160, 160, 160, 192), (160, 160, 192), 192, 192, 'ave', 1, 7))]
        modules += [('inception_f6_4', Inception_B(768, (192, 192, 192, 192, 192), (192, 192, 192), 192, 192, 'ave', 1, 7))]
        modules += [('inception_f6_5', Inception_B(768, (192, 192, 192, 192), (192, 320), 0, 0, 'max', 2, 7))]
        modules += [('inception_f7_1', Inception_C(1280, (448, 384, 384, 384), (384, 384, 384), 192, 320, 'ave', 3))]
        modules += [('inception_f7_2', Inception_C(2048, (448, 384, 384, 384), (384, 384, 384), 192, 320, 'ave', 3))]
        modules += [('linear', L.Linear(2048, category_num))]
        # register layers
        [self.add_link(*link) for link in modules]
        self.modules = modules
        self.name = 'googlenet_v3_{}'.format(category_num)

    def count_parameters(self):
        count = 0
        for name, link in self.modules:
            if 'linear' in name:
                count += functools.reduce(lambda a, b: a * b, link.W.data.shape)
            else:
                count += link.count_parameters()
        return count

    def weight_initialization(self):
        for name, link in self.modules:
            if 'linear' == name:
                self.linear.W.data = self.weight_relu_initialization(self.linear)
                self.linear.b.data = self.bias_initialization(self.linear, constant=0)
            else:
                link.weight_initialization()

    def __call__(self, x, train=True):
        h = self.conv1(x, train)
        h = self.conv2(h, train)
        h = self.conv3(h, train)
        h = F.max_pooling_2d(h, ksize=(3, 3), stride=(2, 2), pad=(1, 1))
        h = self.conv4(h, train)
        h = self.conv5(h, train)
        h = self.conv6(h, train)
        h = self.inception_f5_1(h, train)
        h = self.inception_f5_2(h, train)
        h = self.inception_f5_3(h, train)
        h = self.inception_f6_1(h, train)
        h = self.inception_f6_2(h, train)
        h = self.inception_f6_3(h, train)
        h = self.inception_f6_4(h, train)
        h = self.inception_f6_5(h, train)
        h = self.inception_f7_1(h, train)
        h = self.inception_f7_2(h, train)
        num, categories, y, x = h.data.shape
        # global average pooling
        h = F.reshape(F.average_pooling_2d(h, (y, x)), (num, categories))
        h = F.dropout(h, ratio=0.2, train=train)
        h = self.linear(h)
        return h

    def calc_loss(self, y, t):
        loss = F.softmax_cross_entropy(y, t)
        return loss

    def accuracy(self, y, t, xp=np):
        y.to_cpu()
        t.to_cpu()
        indices = np.where((t.data == np.argmax(y.data, axis=1)) == True)[0]
        accuracy = defaultdict(int)
        for i in indices:
            accuracy[t.data[i]] += 1
        indices = np.where((t.data == np.argmax(y.data, axis=1)) == False)[0]
        false_accuracy = defaultdict(int)
        false_y = np.argmax(y.data, axis=1)
        for i in indices:
            false_accuracy[(t.data[i], false_y[i])] += 1
        return accuracy, false_accuracy
