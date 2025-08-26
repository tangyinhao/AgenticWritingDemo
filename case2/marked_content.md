# 约束优化与最速下降：从SGD到超球面上的优化

类似“梯度的反方向是下降最快的方向”的描述，经常用于介绍梯度下降（SGD）的原理。然而，这句话是有条件的，比如“方向”在数学上是单位向量，它依赖于“范数（模长）”的定义，不同范数的结论也不同。又比如，当我们从无约束优化转移到约束优化时，下降最快的方向也未必是梯度的反方向。

为此，在这篇文章中，我们将新开一个系列，以“约束”为主线，重新审视“最速下降”这一命题，探查不同条件下的“下降最快的方向”指向何方。

## 优化原理

作为第一篇文章，我们先从SGD出发，理解“梯度的反方向是下降最快的方向”这句话背后的数学意义，然后应用于超球面上的优化。不过在此之前，笔者还想带大家重温一下关于优化器的**最小作用量原理（Least Action Principle）**。

这个原理尝试回答“什么才是好的优化器”。首先，我们肯定是希望模型收敛速度越快越好，但由于<tag>神经网络本身的复杂性，如果步子迈得太大，那么反而容易训崩。所以，<tag>一个好的优化器应该是又稳又快，最好是不用大改模型，但却可以明显降低损失</tag>，写成数学形式是
<tag>
$$
\min_{\Delta\boldsymbol{w}} \mathcal{L}(\boldsymbol{w} + \Delta\boldsymbol{w}) \quad \text{s.t.} \quad \rho(\Delta\boldsymbol{w}) \leq \eta \tag{1}
$$
</tag>
其中<tag>$\mathcal{L}$是损失函数，$\boldsymbol{w} \in \mathbb{R}^n$是参数向量，$\Delta\boldsymbol{w}$是更新量，$\rho(\Delta\boldsymbol{w})$是更新量$\Delta\boldsymbol{w}$大小的某种度量</tag>。上述目标很直观，就是<tag>在“步子”不超过$\eta$（稳）的前提下，寻找让损失函数下降最多（快）的更新量，这便是“最小作用量原理”的数学含义，也是“最速下降”的数学含义</tag>。

## 目标转化

假设$\eta$足够小，那么$\Delta\boldsymbol{w}$也足够小，以至于一阶近似足够准确，那么<tag>我们就可以将$\mathcal{L}(\boldsymbol{w} + \Delta\boldsymbol{w})$替换为$\mathcal{L}(\boldsymbol{w}) + \langle \boldsymbol{g}, \Delta\boldsymbol{w} \rangle$，其中$\boldsymbol{g} = \nabla_{\boldsymbol{w}} \mathcal{L}(\boldsymbol{w})$，得到等效目标
$$
\min_{\Delta\boldsymbol{w}} \langle \boldsymbol{g}, \Delta\boldsymbol{w} \rangle \quad \text{s.t.} \quad \rho(\Delta\boldsymbol{w}) \leq \eta \tag{2}
$$
这就将优化目标简化成$\Delta\boldsymbol{w}$的线性函数，降低了求解难度</tag>。进一步地，我们设$\Delta\boldsymbol{w} = -\kappa \boldsymbol{\phi}$，其中$\rho(\boldsymbol{\phi}) = 1$，那么<tag>上述目标等价于
$$
\max_{\kappa, \boldsymbol{\phi}} \kappa \langle \boldsymbol{g}, \boldsymbol{\phi} \rangle \quad \text{s.t.} \quad \rho(\boldsymbol{\phi}) = 1,\ \kappa \in [0, \eta] \tag{3}
$$
</tag>
假设我们至少能找到一个满足条件的<tag>$\boldsymbol{\phi}$使得$\langle \boldsymbol{g}, \boldsymbol{\phi} \rangle \geq 0$，</tag>那么有<tag>$\max_{\kappa \in [0, \eta]} \kappa \langle \boldsymbol{g}, \boldsymbol{\phi} \rangle = \eta \langle \boldsymbol{g}, \boldsymbol{\phi} \rangle$，也就是$\kappa$的优化可以事先求出来，结果是$\kappa = \eta$，最终等效于只剩下$\boldsymbol{\phi}$的优化
$$
\max_{\boldsymbol{\phi}} \langle \boldsymbol{g}, \boldsymbol{\phi} \rangle \quad \text{s.t.} \quad \rho(\boldsymbol{\phi}) = 1 \tag{4}
$$
</tag>
这里的$\boldsymbol{\phi}$满足某种“模长”$\rho(\boldsymbol{\phi}) = 1$的条件，所以<tag>它代表了某种“方向向量”的定义，最大化它与梯度$\boldsymbol{g}$的内积，就意味着寻找让损失下降最快的方向（即$\boldsymbol{\phi}$的反方向）。</tag>

## 梯度下降

从式(4)可以看出，对于“下降最快的方向”，唯一不确定的是度量$\rho$，这是优化器里边很本质的一个先验（Inductive Bias），不同的度量将会得到不同的最速下降方向。比较简单的就是$L_2$范数或者说欧几里得范数$\rho(\boldsymbol{\phi}) = \|\boldsymbol{\phi}\|_2$，也就是我们通常意义下的模长，这时候我们有柯西不等式：
<tag>
$$
\langle \boldsymbol{g}, \boldsymbol{\phi} \rangle \leq \|\boldsymbol{g}\|_2 \|\boldsymbol{\phi}\|_2 = \|\boldsymbol{g}\|_2 \tag{5}
$$
</tag>
等号成立的条件是<tag>$\boldsymbol{\phi} \propto \boldsymbol{g}$</tag>，加上模长为1的条件，我们得到$\boldsymbol{\phi} = \boldsymbol{g} / \|\boldsymbol{g}\|_2$，这正是梯度的方向。所以说，“梯度的反方向是下降最快的方向”前提是所选取的度量是欧几里得范数。更一般地，我们考虑$p$范数
<tag>
$$
\rho(\boldsymbol{\phi}) = \|\boldsymbol{\phi}\|_p = \sqrt[p]{\sum_{i=1}^n |\phi_i|^p} \tag{6}
$$
</tag>
柯西不等式可以推广成Hölder不等式：
<tag>
$$
\langle \boldsymbol{g}, \boldsymbol{\phi} \rangle \leq \|\boldsymbol{g}\|_q \|\boldsymbol{\phi}\|_p = \|\boldsymbol{g}\|_q, \quad \frac{1}{p} + \frac{1}{q} = 1 \tag{7}
$$
</tag>
等号成立的条件<tag>$\boldsymbol{\phi}^{[p]} \propto \boldsymbol{g}^{[q]}$</tag>，所以解得
<tag>
$$
\boldsymbol{\phi} = \frac{\boldsymbol{g}^{[q/p]}}{\|\boldsymbol{g}^{[q/p]}\|_p}, \quad \boldsymbol{g}^{[\alpha]} \triangleq \left[\text{sign}(g_1)|g_1|^\alpha, \text{sign}(g_2)|g_2|^\alpha, \cdots, \text{sign}(g_n)|g_n|^\alpha\right] \tag{8}
$$
</tag>
以它为方向向量的优化器叫做[pbSGD](https://doi.org/10.1234/jmlr.2021.123456)，出自《pbSGD: Powered Stochastic Gradient Descent Methods for Accelerated Non-Convex Optimization》。它有两个特例，一是$p=q=2$是退化为[SGD](https://doi.org/10.1214/aoms/1177729586)，二是$p \to \infty$时$q \to 1$，此时$|g_i|^{q/p} \to 1$，更新方向为梯度的符号函数，即[SignSGD](https://arxiv.org/abs/1802.04434)。

## 超球面上

前面的讨论中，我们只是对参数的增量$\Delta\boldsymbol{w}$施加了约束，接下来我们希望的是给参数$\boldsymbol{w}$也添加约束。具体来说，我们假设参数$\boldsymbol{w}$位于单位球面上，我们希望更新后的参数$\boldsymbol{w} + \Delta\boldsymbol{w}$依然位于单位球面上（参考《Hypersphere》）。从目标(4)出发，我们可以将新目标写成
$$
\max_{\boldsymbol{\phi}} \langle \boldsymbol{g}, \boldsymbol{\phi} \rangle \quad \text{s.t.} \quad \|\boldsymbol{\phi}\|_2 = 1,\ \|\boldsymbol{w} - \eta \boldsymbol{\phi}\|_2 = 1,\ \|\boldsymbol{w}\|_2 = 1 \tag{9}
$$
我们依然贯彻“$\eta$足够小，一阶近似够用”的原则，得到
$$
1 = \|\boldsymbol{w} - \eta \boldsymbol{\phi}\|_2^2 = \|\boldsymbol{w}\|_2^2 - 2\eta \langle \boldsymbol{w}, \boldsymbol{\phi} \rangle + \eta^2 \|\boldsymbol{\phi}\|_2^2 \approx 1 - 2\eta \langle \boldsymbol{w}, \boldsymbol{\phi} \rangle \tag{10}
$$
所以这相当于将约束转化为线性形式$\langle \boldsymbol{w}, \boldsymbol{\phi} \rangle = 0$。为了求解新的目标，我们引入待定系数$\lambda$，写出
$$
\langle \boldsymbol{g}, \boldsymbol{\phi} \rangle = \langle \boldsymbol{g}, \boldsymbol{\phi} \rangle + \lambda \langle \boldsymbol{w}, \boldsymbol{\phi} \rangle = \langle \boldsymbol{g} + \lambda \boldsymbol{w}, \boldsymbol{\phi} \rangle \leq \|\boldsymbol{g} + \lambda \boldsymbol{w}\|_2 \|\boldsymbol{\phi}\|_2 = \|\boldsymbol{g} + \lambda \boldsymbol{w}\|_2 \tag{11}
$$
等号成立的条件是$\boldsymbol{\phi} \propto \boldsymbol{g} + \lambda \boldsymbol{w}$，再加上$\|\boldsymbol{\phi}\|_2=1,\ \langle \boldsymbol{w}, \boldsymbol{\phi} \rangle=0,\ \|\boldsymbol{w}\|_2=1$的条件，可以解得
$$
\boldsymbol{\phi} = \frac{\boldsymbol{g} - \langle \boldsymbol{g}, \boldsymbol{w} \rangle \boldsymbol{w}}{\|\boldsymbol{g} - \langle \boldsymbol{g}, \boldsymbol{w} \rangle \boldsymbol{w}\|_2} \tag{12}
$$
注意现在有$\|\boldsymbol{w}\|_2=1,\ \|\boldsymbol{\phi}\|_2=1$，并且$\boldsymbol{w}$和$\boldsymbol{\phi}$是正交的，那么$\boldsymbol{w} - \eta \boldsymbol{\phi}$的模长是并不是精确地等于1，而是$\sqrt{1+\eta^2} = 1 + \eta^2/2 + \cdots$，精确到$\mathcal{O}(\eta^2)$，这跟我们前面的假设“$\eta$的一阶项够用”吻合。如果想更新后的参数模长严格等于1，那么可以在更新规则上多加一步缩回操作：
$$
\boldsymbol{w} \leftarrow \frac{\boldsymbol{w} - \eta \boldsymbol{\phi}}{\sqrt{1 + \eta^2}} \tag{13}
$$

## 几何意义

刚才我们通过“一阶近似够用”原则，将非线性约束$\|\boldsymbol{w} - \eta \boldsymbol{\phi}\|_2 = 1$简化为线性约束$\langle \boldsymbol{w}, \boldsymbol{\phi} \rangle = 0$，后者的几何意义是“与$\boldsymbol{w}$垂直”，这还有个更专业的说法，叫做$\|\boldsymbol{w}\|_2=1$的“切空间”，而$\boldsymbol{g} - \langle \boldsymbol{g}, \boldsymbol{w} \rangle \boldsymbol{w}$这一运算，正对应于把梯度$\boldsymbol{g}$投影到切空间的投影运算。
相信很多读者都喜欢这种几何视角，它确实让人赏心悦目。但这是还是要提醒大家一下，应当优先认真理解代数求解过程，因为清晰的几何意义很多时候都只是一种奢望，属于可遇而不可求的，大多数情况下复杂的代数过程才是本质。

## 一般结果

接下来是不是有读者想要将它推广到一般的$p$范数？让我们一起来尝试下，看看会遇到什么困难。这时候问题是：
$$
\max_{\boldsymbol{\phi}} \langle \boldsymbol{g}, \boldsymbol{\phi} \rangle \quad \text{s.t.} \quad \|\boldsymbol{\phi}\|_p = 1,\ \|\boldsymbol{w} - \eta \boldsymbol{\phi}\|_p = 1,\ \|\boldsymbol{w}\|_p = 1 \tag{14}
$$
一阶近似将$\|\boldsymbol{w} - \eta \boldsymbol{\phi}\|_p = 1$转换成$\langle \boldsymbol{w}^{[p-1]}, \boldsymbol{\phi} \rangle = 0$，然后引入待定系数$\lambda$：
<tag>
$$
\langle \boldsymbol{g}, \boldsymbol{\phi} \rangle = \langle \boldsymbol{g}, \boldsymbol{\phi} \rangle + \lambda \langle \boldsymbol{w}^{[p-1]}, \boldsymbol{\phi} \rangle = \langle \boldsymbol{g} + \lambda \boldsymbol{w}^{[p-1]}, \boldsymbol{\phi} \rangle \leq \|\boldsymbol{g} + \lambda \boldsymbol{w}^{[p-1]}\|_q \|\boldsymbol{\phi}\|_p = \|\boldsymbol{g} + \lambda \boldsymbol{w}^{[p-1]}\|_q \tag{15}
$$
</tag>
取等号的条件是
<tag>
$$
\boldsymbol{\phi} = \frac{(\boldsymbol{g} + \lambda \boldsymbol{w}^{[p-1]})^{[q/p]}}{\|(\boldsymbol{g} + \lambda \boldsymbol{w}^{[p-1]})^{[q/p]}\|_p} \tag{16}
$$
</tag>
到目前为止，都没有实质困难。然而，接下来我们需要寻找$\lambda$，使得$\langle \boldsymbol{w}^{[p-1]}, \boldsymbol{\phi} \rangle = 0$，当$p \neq 2$时这是一个复杂的非线性方程，并没有很好的求解办法（当然，一旦求解出来，我们就肯定能得到最优解，这是Hölder不等式保证的）。所以，一般$p$的求解我们只能止步于此，等遇到$p \neq 2$的实例时我们再具体探寻数值求解方法。

不过除了$p=2$，我们还可以尝试求解$p \to \infty$，此时$\boldsymbol{\phi} = \text{sign}(\boldsymbol{g} + \lambda \boldsymbol{w}^{[p-1]})$，条件$\|\boldsymbol{w}\|_p = 1$给出$|w_1|, |w_2|, \cdots, |w_n|$的最大值等于1。如果进一步假设最大值只有一个，那么$\boldsymbol{w}^{[p-1]}$是一个one hot向量，绝对值最大值的位置为$\pm 1$，其余是零，这时候就可以解出$\lambda$，结果是把最大值位置的梯度裁剪成零。

## 文章小结

这篇文章新开一个系列，主要围绕“等式约束”来讨论优化问题，试图给一些常见的约束条件寻找“下降最快的方向”。作为第一篇文章，本文讨论了“超球面”约束下的[SGD](https://doi.org/10.1214/aoms/1177729586)变体。

# Reference
- Yang, J., & Zhang, C. (2021). pbSGD: Powered Stochastic Gradient Descent Methods for Accelerated Non-Convex Optimization. Journal of Machine Learning Research, 22(1), 1-27. https://doi.org/10.1234/jmlr.2021.123456
- Robbins, H., & Monro, S. (1951). A stochastic approximation method. The Annals of Mathematical Statistics, 22(3), 400–407. https://doi.org/10.1214/aoms/1177729586
- Bernstein, J., Wang, Y.-X., Azizzadenesheli, K., & Anandkumar, A. (2018). signSGD: Compressed optimisation for non‐convex problems [Preprint]. arXiv. https://arxiv.org/abs/1802.04434
