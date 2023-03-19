# 神经网络

### 代价函数

## 反向传播算法(Backpropagation algorithm)

```
L: 总层数
l: 第l层
j: 第l+1层第j个节点
i: 第l层第i个节点
δ: 偏置项(bias),实质是代价函数删去正则化项后关于神经元z的偏导
Δ: δ(delta)的大写
Θ: 权重，相当于θ(theta)
J: 代价函数
D: 为了计算代价函数J的偏导
```

#### 步骤(原版)

* **读入数据**
* **令$Δ^{(l)}_{ji}=0$**(所有l,i,j)(初始化delta)
* 对所有在train中的数据$x^{(1)}-x^{(m)}$计算
* 令$a^{(1)}=x^{(k)}$
* 计算$a^{(l)}_{ji}$
* 用$y^{(i)}$计算最后一层$δ^{(L)}=a^{(L)}-y^{(i)}$ `向量`
* 从最后一层向前传播计算$δ^{(L-1)},δ^{(L-2)},...,δ^{(2)}$
  1. `计算公式:`$δ^{(l)}=(Θ^{(l)})^Tδ^{(l+1)}.*g'(z^{(l)})$
  2. $g'(z^{(l)})=a^{(l)}.*(1-a^{(l)})$
* $Δ^{(l)}:=Δ^{(l)}+δ^{(l+1)}(a^{(l)})^T$`一次赋值一层`
* $D^{(l)}_{ji}=\frac{1}{m}Δ^{(l)}_{ji}+\frac{1}{λ}Θ^{(l)}_{ji}$ `i!=0`
* $D^{(l)}_{ji}=\frac{1}{m}Δ^{(l)}_{ji}$ `i==0`
  `理解:`$\frac{∂}{∂Θ^{(l)}_{ji}}J(Θ)=D^{(l)}_{ji}$

### 梯度检测

```
用导数定义计算值，判断和用反向传播算法计算出来的值差别大不大，大的话就出bug了（写错了
```

### 随机初始化

```
如果初始化权重成一样的会导致a分不开，每次梯度下降后权重还是一样，出bug了
```

用rand函数之类的初始化

### 小细节

* 最后一层的激活函数只能用sigmoid或者softmax
* 用ReLU学习率不能设置太大 不然会炸(数据意义上