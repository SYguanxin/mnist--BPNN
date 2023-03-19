<!-- TOC -->
* [pandas](#pandas)
* [numpy](#numpy)
* [~~折磨~~](#)
* [最后结果](#)
<!-- TOC -->
[主资料](https://blog.csdn.net/weixin_43499292/article/details/120570293)


> [python-如何使用numpy将label转为one-hot](https://blog.csdn.net/weixin_41177022/article/details/124330994)  
`label = np.eye(10, dtype=np.uint8)[label]`  # 获取label的onehot编码矩阵

#### pandas 
> DataFrame提取列  
`X = np.asarray(train.loc[:, train.columns[1:]])`  # 特征读取

#### numpy  
* 合并两个矩阵 `X = np.hstack((x0, X))` `np.vstack()`
* 合并俩向量 `X = np.concatenate(v1, v2)`

## ~~折磨~~
* 3.13
写根据输入层数自适应层数的BPNN
试下没加特征缩放结果exp()之后太大了

* 3.14
尝试直接用矩阵把所有数据一次计算完

* 3.15
时间快不够了，正常做吧）之后再改
用增加偏置节点的办法，要特判正则化项好麻烦，~~一起加了吧~~）
不对，想想可以直接把正则化项矩阵第一列置0  
写的代码没用框架硬件利用率低导致一次迭代5分钟）
改用小批量梯度下降

* 3.16
?????????
准确率88不动了
尝试梯度检测

* 3.17
???
我改了什么
准确率自动上升到92
迭代10000次试试
!!!94准确率

* 3.18
用ReLU试试
(准确率10%，print梯度全是nan
原来最后一层的激活函数不能用ReLU
过拟合了(悲
训练集准确率96，测试集准确率94

* 3.19
试着加了一层3/2
训练集准确率99.9 测试集95.2
好离谱的过拟合  
准确度96.8不写了
想写卷积神经网络

# 最后结果
群里test.csv准确率95.8%  
答案放在./datafile/answer.csv