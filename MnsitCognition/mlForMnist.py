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
        
        # 所有层的输出
        self.Alloutputs = []
        # 除输入层外的所有层的误差
        self.AllErrors = []

        # 初始化权重  0 是输入到  第一个隐藏层的；  1是  最后一个隐藏层到输出层的权重
        self.wAll = []
        # 第一层
        self.wAll.append(numpy.random.normal(0.0, pow(self.iNodes, -0.5), (self.hNodes[0], self.iNodes)))

        # 隐藏层超过1个
        if (len(self.hNodes) > 1):
            self.hMOneFlag = 1
1
        # 中间的隐藏层之间权重初始化
        if (self.hMOneFlag == 1):
          for i in range(1, len( self.hNodes)-1):
              self.wAll.append(numpy.random.normal(0.0, pow(self.hNodes[i-1], -0.5), (self.hNodes[i], self.hNodes[i-1]))

        # 隐藏层 到 输出层的权重
        self.wAll.append(numpy.random.normal(0.0, pow(self.hNodes[len( self.hNodes) -1], -0.5), ( self.oNodes, self.hNodes[len( self.hNodes) -1]  )))

        # 初始化 激活函数
        self.activation_function = lambda x: scipy.special.expit(x)

    def train(self, inputs_list, targets_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        
        # 正向传播的  实际值
        final_outputs = self.forSpresd(inputs)
        
         # output layer error is the (target - actual)
        output_errors = targets - final_outputs
        # 反馈传播，更新各层的权重值
        self.updataWAll( output_errors)
        
        
    def forSpresd(self, inputs)
        '''
            执行一次向前传播
            inputs ： 本次向前的输入值
        '''
        # 每次训练的各层的输出不同，需初始化
        self.Alloutputs = []
        
        # 输入层（第一层）的输出
        self.Alloutputs.append(inputs)
、
        # 第一层隐藏层的输入
        hidden_inputs_1 = numpy.dot(self.wAll[0], inputs)

        # 第一个隐藏层的输入
        inputs_temp = hidden_inputs_1
        
        if (self.hMOneFlag == 1):
            # 得到 最后一隐藏层的输出
            for i in range(1, (len(self.hNodes)-1)  ):
                # 一个隐藏层的输出
                outputs_temp = self.activation_function(inputs_temp)
                # 加入到各层的输出集合
                self.Alloutputs,append(outputs_temp)
                # 下一个隐藏层的输入
                inputs_temp = numpy.dot(self.wAll[i], outputs_temp)
    
        # 最后一隐藏层的输入为inputs_temp，可得最后一隐藏层的输出
        h_final_out = self.activation_function(inputs_temp)
        # 最后一隐藏层的输出
        self.Alloutputs.append(h_final_out)
        
        # calculate signals into final output layer
        final_inputs = numpy.dot(self.wAll[len(self.hNodes)], h_final_out)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        # 输出层的输出
        self.Alloutputs.append(final_outputs)
        
        return final_outputs
       
    def updateWAll(self, output_errors)
    
        self.AllErrors = []
        # 存储除输入层的其他层  误差
        self.AllErrors.append(output_errors)

        next_error = output_errors
        for i in range( len(self.hNodes), 1):
            this_error = numpy.dot(self.wAll[i].T, next_error)
            # 插在列表头
            self.AllErrors.insert(0,this_error)
            next_error = this_error

        # 更新权重
        for i in range(0, len(self.hNodes))
            self.wAll[i] += self.lr * numpy.dot((self.AllErrors[i] * self.Alloutputs[i+1] * (1.0 - self.Alloutputs[i+1])), numpy.transpose(self.Alloutputs[i]))
            
    # query the neural network
    def query(self, inputs_list):
         inputs = numpy.array(inputs_list, ndmin=2).T
         return self.forSpresd(inputs)