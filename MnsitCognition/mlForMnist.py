import  numpy

import scipy.special


import matpotlib.pyplot


class neuralNetwork:

    # 初始化 网络
    def __init__(self, inputNodes, hiddenNodes, outputNodes, learningrating ):


        # 初始化网络的层数
        self.iNodes = inputNodes
        # 隐藏层数组，第一个元素对应第一个隐藏层的节点数，方便扩展  多层隐藏层
        self.hNodes = hiddenNodes
        self.oNodes = outputNodes
        self.lr= learningrating

        # 初始化权重
        self.wAll = []
        # 第一层
        self.wAll.append(numpy.random.normal(0.0, pow(self.iNodes, -0.5), (self.hNodes[0], self.iNodes)))

        # 隐藏层超过1个
        if (len(self.hNodes) > 1):
            self.hOneFlag = 1

        # 中间的隐藏层之间权重初始化
        if (self.hOneFlag == 1):
          for i in range(1, len( self.hNodes)-1):
              self.wAll.append(numpy.random.normal(0.0, pow(self.hNodes[i-1], -0.5), (self.hNodes[i], self.hNodes[i-1]))

        # 隐藏层 到 输出层的权重
        self.wAll.append(numpy.random.normal(0.0, pow(self.hNodes[len( self.hNodes) -1], -0.5), ( self.oNodes, self.hNodes[len( self.hNodes) -1]  )))

        # 初始化 激活函数
        self.activation_function = lambda x: scipy.special.expit(x)