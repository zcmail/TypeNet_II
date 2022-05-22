# TypeNet_II

#### 介绍
电子与信息学报投稿：一种可解释的自由文本击键事件序列分类模型.
<<<<<<< HEAD
An Interpretable Free-text Keystroke Event Sequence Classification Model.
TypeNet is a Siamese network model based on two-layer LSTM branch structure. It has achieved good results in the classification of free-text keystroke event sequences, but lacks interpretation. Therefore, the TypeNet model was transformed, and a Siamese network TypeNet II based on a single-layer LSTM branch structure was proposed. A multi-layer perceptron is used to measure the similarity of two feature sequences reflected by the absolute value of the difference between the output embeddings of the two branches. After the model training, the multi-layer perceptron is simulated by a multivariate binomial expression. Based on the obtained multivariate binomial expression, the classification judgment of the model can be explained. The experimental results show that the classification effect of the TypeNet II model exceeds the existing TypeNet model. The results of multivariate binomial regression are generalized, and there is a nonlinear relationship between the absolute value of the difference of the embeddings and the similarity measure.

#### 软件架构
Python3.7.6
keras2.4.3

#### 使用说明
1.TypeNet文件夹下是复现及分析TypeNet（TypeNet: Deep Learning Keystroke Biometrics[J]. arXiv:2101.05570）.
2.TypeNet_II文件夹下是“一种可解释的自由文本击键事件序列分类模型”的相关工作.
3.原始数据来自（Vivek Dhakal，Anna Maria Feit，Per Ola Kristensson，Antti Oulasvirta. Observations on Typing from 136 Million Keystrokes[C]. CHI 2018, Montreal, QC, Canada, April 21-26, 2018.），这里按TypeNet文献中特征序列组织形式对原始数据进行了转换：train_five_tuple_vector_data_del_null.json是训练用的数据（6万多被试者），test_five_tuple_vector_data_del_null.json是测试用的数据（10万多被试者），5_subject_for_test.json是从test数据中随机选了5个被试者的数据，用于代码调试和可视化分析。
4.训练模型的代码在.py文件中，分析模型的代码在.ipynb文件中。数据文件在百度网盘：https://pan.baidu.com/s/10AMybivmY6PL5yCkv5MAVQ 
提取码：nxqx。

#### 参与贡献
1.zhang_chang_xd@163.com
=======