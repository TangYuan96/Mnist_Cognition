import  numpy

import scipy.special


# import matpotlib.pyplot


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

        # 隐藏层超过两层的标签
        self.hMOneFlag = 0
        # 初始化权重  0 是输入到  第一个隐藏层的；  1是  最后一个隐藏层到输出层的权重
        self.wAll = []
        # 第一层
        self.wAll.append(numpy.random.normal(0.0, pow(self.iNodes, -0.5), (self.hNodes[0], self.iNodes)))

        # 隐藏层超过1个
        if (len(self.hNodes) > 1):
            self.hMOneFlag = 1

        # 中间的隐藏层之间权重初始化
        if ((self.hMOneFlag) == 1):
          for i in range(1, len( self.hNodes)):
              self.wAll.append(numpy.random.normal(0.0, pow(self.hNodes[i-1], -0.5), (self.hNodes[i], self.hNodes[i-1])))

        # 隐藏层 到 输出层的权重
        self.wAll.append(numpy.random.normal(0.0, pow(self.hNodes[len( self.hNodes) -1], -0.5), ( self.oNodes, self.hNodes[len( self.hNodes) -1]  )))

        # 初始化 激活函数
        self.activation_function = lambda x: scipy.special.expit(x)

    def updateWAll(self, output_errors):

        self.AllErrors = []
        # 存储除输入层的其他层  误差
        self.AllErrors.append(output_errors)

        next_error = output_errors
        for i in range(len(self.hNodes), 0,-1):
            this_error = numpy.dot(self.wAll[i].T, next_error)
            # 插在列表头

            self.AllErrors.insert(0, this_error)
            next_error = this_error

        # 更新权重
        for i in range(0, len(self.hNodes)+1):
            self.wAll[i] += self.lr * numpy.dot( (self.AllErrors[i] * self.Alloutputs[i + 1] * (1.0 - self.Alloutputs[i + 1])), numpy.transpose(self.Alloutputs[i]))

    def train(self, inputs_list, targets_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        
        # 正向传播的  实际值
        final_outputs = self.forSpresd(inputs)
        
         # output layer error is the (target - actual)
        output_errors = targets - final_outputs
        # 反馈传播，更新各层的权重值
        self.updateWAll( output_errors)
        
        
    def forSpresd(self, inputs):
        '''
            执行一次向前传播
            inputs ： 本次向前的输入值
        '''
        # 每次训练的各层的输出不同，需初始化
        self.Alloutputs = []
        
        # 输入层（第一层）的输出
        self.Alloutputs.append(inputs)

        # 第一层隐藏层的输入
        hidden_inputs_1 = numpy.dot(self.wAll[0], inputs)

        # 第一个隐藏层的输入
        inputs_temp = hidden_inputs_1
        #
        # if (self.hMOneFlag == 1):
            # 得到 最后一隐藏层的输出
        # print("train  h")

        for i in range(1, (len(self.hNodes)+1)  ):
            # 一个隐藏层的输出
            outputs_temp = self.activation_function(inputs_temp)
            # 加入到各层的输出集合
            self.Alloutputs.append(outputs_temp)
            # 下一个隐藏层的输入
            inputs_temp = numpy.dot(self.wAll[i], outputs_temp)
    
        # 此处的 inputs_temp 为输出层的 输入信号

        # h_final_out = self.activation_function(inputs_temp)
        # print("  h_final_out:",h_final_out.shape)
        # # 最后一隐藏层的输出
        # self.Alloutputs.append(h_final_out)
        #
        # # calculate signals into final output layer
        # final_inputs = numpy.dot(self.wAll[len(self.hNodes)], h_final_out)


        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(inputs_temp)
        # 输出层的输出
        self.Alloutputs.append(final_outputs)
        
        return final_outputs
       

            
    # query the neural network
    def query(self, inputs_list):
         inputs = numpy.array(inputs_list, ndmin=2).T
         return self.forSpresd(inputs)



if __name__ == "__main__":


    # 网络初始化
    # number of input, hidden and output nodes
    input_nodes = 784
    hidden_nodes = [200]
    output_nodes = 10

    # learning rate
    learning_rate = 0.1

    # create instance of neural network
    n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)


    # 进行  训练
    # load the mnist training data CSV file into a list
    training_data_file = open("MnIST sets/mnist_train.csv", 'r')
    training_data_list = training_data_file.readlines()
    training_data_file.close()

    # train the neural network

    # epochs is the number of times the training data set is used for training
    epochs = 2


    print(".......start train......")
    for e in range(epochs):
        # go through all records in the training data set

        print("\n.... %dth训练",e+1)
        for record in training_data_list:
            # split the record by the ',' commas
            all_values = record.split(',')
            # scale and shift the inputs
            inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            # create the target output values (all 0.01, except the desired label which is 0.99)
            targets = numpy.zeros(output_nodes) + 0.01
            # all_values[0] is the target label for this record
            targets[int(all_values[0])] = 0.99
            n.train(inputs, targets)
            pass
        pass



    print("\n........start test.....")
    # 进行测试
    # load the mnist test data CSV file into a list
    test_data_file = open("MnIST sets/mnist_test.csv", 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    # test the neural network

    # scorecard for how well the network performs, initially empty
    scorecard = []

    # go through all the records in the test data set
    for record in test_data_list:
        # split the record by the ',' commas
        all_values = record.split(',')
        # correct answer is first value
        correct_label = int(all_values[0])
        # scale and shift the inputs
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # query the network
        outputs = n.query(inputs)
        # the index of the highest value corresponds to the label
        label = numpy.argmax(outputs)
        # append correct or incorrect to list
        if (label == correct_label):
            # network's answer matches correct answer, add 1 to scorecard
            scorecard.append(1)
        else:
            # network's answer doesn't match correct answer, add 0 to scorecard
            scorecard.append(0)
            pass

        pass

    # calculate the performance score, the fraction of correct answers
    scorecard_array = numpy.asarray(scorecard)
    print("performance = ", scorecard_array.sum() / scorecard_array.size)
