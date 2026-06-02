# GroupNorm<a name="ZH-CN_TOPIC_0000002554343729"></a>

## 产品支持情况<a name="section1586581915393"></a>

<a name="table169596713360"></a>
<table><thead align="left"><tr id="row129590715369"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p17959971362"><a name="p17959971362"></a><a name="p17959971362"></a><span id="ph895914718367"><a name="ph895914718367"></a><a name="ph895914718367"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p89594763612"><a name="p89594763612"></a><a name="p89594763612"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row18959673369"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p1595910763613"><a name="p1595910763613"></a><a name="p1595910763613"></a><span id="ph1595918753613"><a name="ph1595918753613"></a><a name="ph1595918753613"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p1695957133611"><a name="p1695957133611"></a><a name="p1695957133611"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section24260390593"></a>

对一个特征进行标准化的一般公式如下所示：

<!-- img2text -->
$$x_i'=\frac{x_i-\mu}{\sigma}$$

其中，i表示特征中的索引，![](figures/zh-cn_formulaimage_0000002523345896.png)  和  ![](figures/zh-cn_formulaimage_0000002523305920.png)  表示特征中每个值标准化前后的值，μ和σ表示特征的均值和标准差，计算公式如下所示：

<!-- img2text -->
$$
\mu=\frac{1}{m}\sum_{i\in S}x_i,\quad \sigma=\sqrt{\frac{1}{m}\sum_{i\in S}(x_i-\mu)^2+\epsilon}
$$

<!-- img2text -->
$$
\mu = \frac{1}{m}\sum_{i \in S} x_i,\quad \sigma = \sqrt{\frac{1}{m}\sum_{i \in S}(x_i - \mu)^2 + \epsilon}
$$

其中，ε是一个很小的常数，S表示参与计算的数据的集合，m表示集合的大小。不同类型的特征标准化方法（BatchNorm、LayerNorm、InstanceNorm、GroupNorm等）的主要区别在于参与计算的数据集合的选取上。不同Norm类算子参与计算的数据集合的选取方式如下：

<!-- img2text -->
```text
Batch Norm                    Layer Norm                    Instance Norm                 Group Norm

         H, W                          H, W                          H, W                         H, W
          ↑                             ↑                             ↑                            ↑
          │                             │                             │                            │

      ┌─────────┐                   ┌─────────┐                   ┌─────────┐                  ┌─────────┐
     /░░░░░░░░░/|                  /░░░░░░░░░/|                  /░░░░░░░░░/|                 /░░░░░░░░░/|
    /░░░░░░░░░/ |                 /░░░░░░░░░/ |                 /░░░░░░░░░/ |                /░░░░░░░░░/ |
   /░░░░░░░░░/  |                /████░░░░░/  |                /░░░░░░░░░/  |               /░░░░░░░░░/  |
  ┌─────────┐   |               ┌─────────┐   |               ┌─────────┐   |              ┌─────────┐   |
  │░░░░█████│   |               │████░░░░░│   |               │░░░██░░░░│   |              │░░███░░░░│   |
  │░░░░█████│   |               │████░░░░░│   |               │░░░██░░░░│   |              │░░███░░░░│   |
  │░░░░█████│   |               │████░░░░░│   |               │░░░██░░░░│   |              │░░███░░░░│   |
  │░░░░█████│   ├─ N            │████░░░░░│   ├─ N            │░░░██░░░░│   ├─ N           │░░███░░░░│   ├─ N
  │░░░░█████│  /                │████░░░░░│  /                │░░░██░░░░│  /               │░░███░░░░│  /
  │░░░░█████│ /                 │████░░░░░│ /                 │░░░██░░░░│ /                │░░███░░░░│ /
  └─────────┘/                  └─────────┘/                  └─────────┘/                 └─────────┘/
      C                              C                              C                             C
```

说明:
- 图中阴影/高亮区域表示参与计算的数据集合 S。
- Batch Norm：按 N 维上的一个切片进行归一化示意。
- Layer Norm：按单个样本内的整层（C, H, W）进行归一化示意。
- Instance Norm：按单个样本的单个通道在 H, W 范围内进行归一化示意。
- Group Norm：按单个样本中 C 维分组后，每组在对应的 H, W 范围内进行归一化示意。

对于一个shape为\[N, C, H, W\]的输入，GroupNorm将每个\[C, H, W\]在C维度上分为groupNum组，然后对每一组进行标准化。最后对标准化后的特征进行缩放和平移。其中缩放参数γ和平移参数β是可训练的。

<!-- img2text -->
$$
Y = \frac{X - E[X]}{\sqrt{\operatorname{Var}[X] + \epsilon}} \times \gamma + \beta
$$

## 函数原型<a name="section1320612264404"></a>

-   接口框架申请临时空间

    ```
    template <typename T, bool isReuseSource = false>
    __aicore__ inline void GroupNorm(const LocalTensor<T>& output, const LocalTensor<T>& outputMean, const LocalTensor<T>& outputVariance, const LocalTensor<T>& inputX, const LocalTensor<T>& gamma, const LocalTensor<T>& beta, const T epsilon, GroupNormTiling& tiling)
    ```

-   通过sharedTmpBuffer入参传入临时空间

    ```
    template <typename T, bool isReuseSource = false>
    __aicore__ inline void GroupNorm(const LocalTensor<T>& output, const LocalTensor<T>& outputMean, const LocalTensor<T>& outputVariance, const LocalTensor<T>& inputX, const LocalTensor<T>& gamma, const LocalTensor<T>& beta, const LocalTensor<uint8_t>& sharedTmpBuffer, const T epsilon, GroupNormTiling& tiling)
    ```

## 参数说明<a name="section164281039154020"></a>

**表 1**  模板参数说明

<a name="table575571914269"></a>
<table><thead align="left"><tr id="row18755131942614"><th class="cellrowborder" valign="top" width="19.39%" id="mcps1.2.3.1.1"><p id="p675519193268"><a name="p675519193268"></a><a name="p675519193268"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="80.61%" id="mcps1.2.3.1.2"><p id="p375511918267"><a name="p375511918267"></a><a name="p375511918267"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row14755141911264"><td class="cellrowborder" valign="top" width="19.39%" headers="mcps1.2.3.1.1 "><p id="p47551198266"><a name="p47551198266"></a><a name="p47551198266"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="80.61%" headers="mcps1.2.3.1.2 "><p id="p125969172719"><a name="p125969172719"></a><a name="p125969172719"></a>操作数的数据类型。</p>
<p id="p8154124181311"><a name="p8154124181311"></a><a name="p8154124181311"></a><span id="ph9648123613427"><a name="ph9648123613427"></a><a name="ph9648123613427"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：half、float。</p>
</td>
</tr>
<tr id="row6356241194912"><td class="cellrowborder" valign="top" width="19.39%" headers="mcps1.2.3.1.1 "><p id="p143561041144915"><a name="p143561041144915"></a><a name="p143561041144915"></a>isReuseSource</p>
</td>
<td class="cellrowborder" valign="top" width="80.61%" headers="mcps1.2.3.1.2 "><p id="p1275717613718"><a name="p1275717613718"></a><a name="p1275717613718"></a>是否允许修改源操作数，默认值为false。如果开发者允许源操作数被改写，可以使能该参数，使能后能够节省部分内存空间。</p>
<p id="p175786163713"><a name="p175786163713"></a><a name="p175786163713"></a>设置为<strong id="b575706193714"><a name="b575706193714"></a><a name="b575706193714"></a>true</strong>，则本接口内部计算时<strong id="b147578614378"><a name="b147578614378"></a><a name="b147578614378"></a>复用</strong>inputX的内存空间，节省内存空间；设置为<strong id="b475717616379"><a name="b475717616379"></a><a name="b475717616379"></a>false</strong>，则本接口内部计算时<strong id="b157575653719"><a name="b157575653719"></a><a name="b157575653719"></a>不复用</strong>inputX的内存空间。</p>
<p id="p177571162377"><a name="p177571162377"></a><a name="p177571162377"></a>对于float数据类型的输入支持开启该参数，half数据类型的输入不支持开启该参数。</p>
<p id="p62891018544"><a name="p62891018544"></a><a name="p62891018544"></a>isReuseSource的使用样例请参考<a href="更多样例-104.md#section639165323915">更多样例</a>。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  接口参数说明

<a name="zh-cn_topic_0235751031_table33761356"></a>
<table><thead align="left"><tr id="zh-cn_topic_0235751031_row27598891"><th class="cellrowborder" valign="top" width="17.77%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="9.69%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="72.54%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row42461942101815"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p8796143175319"><a name="p8796143175319"></a><a name="p8796143175319"></a>output</p>
</td>
<td class="cellrowborder" valign="top" width="9.69%" headers="mcps1.2.4.1.2 "><p id="p186293346150"><a name="p186293346150"></a><a name="p186293346150"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="72.54%" headers="mcps1.2.4.1.3 "><p id="p58791759185415"><a name="p58791759185415"></a><a name="p58791759185415"></a>目的操作数，对标准化后的输入进行缩放和平移计算的结果。shape为[N, C, H, W]。</p>
<p id="p16911647191712"><a name="p16911647191712"></a><a name="p16911647191712"></a><span id="zh-cn_topic_0000002523303824_ph173308471594"><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><span id="zh-cn_topic_0000002523303824_ph9902231466"><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816"><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
</td>
</tr>
<tr id="row250163914199"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p48809611553"><a name="p48809611553"></a><a name="p48809611553"></a>outputMean</p>
</td>
<td class="cellrowborder" valign="top" width="9.69%" headers="mcps1.2.4.1.2 "><p id="p1662903414157"><a name="p1662903414157"></a><a name="p1662903414157"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="72.54%" headers="mcps1.2.4.1.3 "><p id="p154129195515"><a name="p154129195515"></a><a name="p154129195515"></a>目的操作数，均值。shape为[N, groupNum]。</p>
<p id="p14442002515"><a name="p14442002515"></a><a name="p14442002515"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_1"><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_1"><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_1"><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
</td>
</tr>
<tr id="row041413392198"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p4630634141515"><a name="p4630634141515"></a><a name="p4630634141515"></a>outputVariance</p>
</td>
<td class="cellrowborder" valign="top" width="9.69%" headers="mcps1.2.4.1.2 "><p id="p263018345154"><a name="p263018345154"></a><a name="p263018345154"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="72.54%" headers="mcps1.2.4.1.3 "><p id="p16656103755517"><a name="p16656103755517"></a><a name="p16656103755517"></a>目的操作数，方差。shape为[N, groupNum]。</p>
<p id="p685002019514"><a name="p685002019514"></a><a name="p685002019514"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_2"><a name="zh-cn_topic_0000002523303824_ph173308471594_2"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_2"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_2"><a name="zh-cn_topic_0000002523303824_ph9902231466_2"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_2"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_2"><a name="zh-cn_topic_0000002523303824_ph1782115034816_2"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_2"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
</td>
</tr>
<tr id="row189131716562"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p14803121835620"><a name="p14803121835620"></a><a name="p14803121835620"></a>inputX</p>
</td>
<td class="cellrowborder" valign="top" width="9.69%" headers="mcps1.2.4.1.2 "><p id="p9961712566"><a name="p9961712566"></a><a name="p9961712566"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.54%" headers="mcps1.2.4.1.3 "><p id="p314165011159"><a name="p314165011159"></a><a name="p314165011159"></a>源操作数。shape为[N, C, H, W]。</p>
<p id="p107614424512"><a name="p107614424512"></a><a name="p107614424512"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_3"><a name="zh-cn_topic_0000002523303824_ph173308471594_3"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_3"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_3"><a name="zh-cn_topic_0000002523303824_ph9902231466_3"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_3"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_3"><a name="zh-cn_topic_0000002523303824_ph1782115034816_3"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_3"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
</td>
</tr>
<tr id="row927571485616"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p1727531425614"><a name="p1727531425614"></a><a name="p1727531425614"></a>gamma</p>
</td>
<td class="cellrowborder" valign="top" width="9.69%" headers="mcps1.2.4.1.2 "><p id="p1927518141563"><a name="p1927518141563"></a><a name="p1927518141563"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.54%" headers="mcps1.2.4.1.3 "><p id="p12048552562"><a name="p12048552562"></a><a name="p12048552562"></a>源操作数，缩放参数。该参数支持的取值范围为[-100, 100]。shape为[C]。</p>
<p id="p161986695217"><a name="p161986695217"></a><a name="p161986695217"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_4"><a name="zh-cn_topic_0000002523303824_ph173308471594_4"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_4"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_4"><a name="zh-cn_topic_0000002523303824_ph9902231466_4"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_4"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_4"><a name="zh-cn_topic_0000002523303824_ph1782115034816_4"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_4"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
</td>
</tr>
<tr id="row14907103561"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p12490410195613"><a name="p12490410195613"></a><a name="p12490410195613"></a>beta</p>
</td>
<td class="cellrowborder" valign="top" width="9.69%" headers="mcps1.2.4.1.2 "><p id="p3490610195612"><a name="p3490610195612"></a><a name="p3490610195612"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.54%" headers="mcps1.2.4.1.3 "><p id="p560622519572"><a name="p560622519572"></a><a name="p560622519572"></a>源操作数，平移参数。该参数支持的取值范围为[-100, 100]。shape为[C]。</p>
<p id="p969163045210"><a name="p969163045210"></a><a name="p969163045210"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_5"><a name="zh-cn_topic_0000002523303824_ph173308471594_5"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_5"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_5"><a name="zh-cn_topic_0000002523303824_ph9902231466_5"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_5"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_5"><a name="zh-cn_topic_0000002523303824_ph1782115034816_5"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_5"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
</td>
</tr>
<tr id="row1421513531574"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p102155531574"><a name="p102155531574"></a><a name="p102155531574"></a>sharedTmpBuffer</p>
</td>
<td class="cellrowborder" valign="top" width="9.69%" headers="mcps1.2.4.1.2 "><p id="p5654083582"><a name="p5654083582"></a><a name="p5654083582"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.54%" headers="mcps1.2.4.1.3 "><p id="p104071111204211"><a name="p104071111204211"></a><a name="p104071111204211"></a>接口内部复杂计算时用于存储中间变量，由开发者提供。</p>
<p id="p493111815817"><a name="p493111815817"></a><a name="p493111815817"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_6"><a name="zh-cn_topic_0000002523303824_ph173308471594_6"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_6"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_6"><a name="zh-cn_topic_0000002523303824_ph9902231466_6"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_6"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_6"><a name="zh-cn_topic_0000002523303824_ph1782115034816_6"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_6"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p5881016172817"><a name="p5881016172817"></a><a name="p5881016172817"></a>临时空间大小BufferSize的获取方式请参考<a href="GroupNorm-Tiling.md">GroupNorm Tiling</a>。</p>
</td>
</tr>
<tr id="row88897586577"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p18889155835712"><a name="p18889155835712"></a><a name="p18889155835712"></a>epsilon</p>
</td>
<td class="cellrowborder" valign="top" width="9.69%" headers="mcps1.2.4.1.2 "><p id="p108890588578"><a name="p108890588578"></a><a name="p108890588578"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.54%" headers="mcps1.2.4.1.3 "><p id="p854221075919"><a name="p854221075919"></a><a name="p854221075919"></a>防除0的权重系数。数据类型需要与inputX/output保持一致。</p>
</td>
</tr>
<tr id="row188735226595"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p0277924155919"><a name="p0277924155919"></a><a name="p0277924155919"></a>tiling</p>
</td>
<td class="cellrowborder" valign="top" width="9.69%" headers="mcps1.2.4.1.2 "><p id="p78736227591"><a name="p78736227591"></a><a name="p78736227591"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.54%" headers="mcps1.2.4.1.3 "><p id="p133169331592"><a name="p133169331592"></a><a name="p133169331592"></a>输入数据的切分信息，Tiling信息的获取请参考<a href="GroupNorm-Tiling.md">GroupNorm Tiling</a>。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section823818495405"></a>

无

## 约束说明<a name="section2145132110200"></a>

-   操作数地址对齐要求请参见[通用地址对齐约束](通用说明和约束.md#section796754519912)。
-   当前仅支持ND格式的输入，不支持其他格式。

## 调用示例<a name="section552514954111"></a>

完整的调用样例可参考[GroupNorm样例](https://gitcode.com/cann/asc-devkit/tree/master/examples/03_libraries/03_normalization/groupnorm)。

```
// output: 存放 GroupNorm 计算结果的 Tensor
// outputMean: 输出每个 group 的均值
// outputVariance: 输出每个 group 的方差
// inputX: 输入数据X，shape 为 [N, C, H, W]
// gamma: LayerNorm 的缩放参数 γ，shape 为 [C]
// beta: LayerNorm 的偏置参数 β，shape 为 [C]
// epsilon: 防除零系数ε
// tiling: 预计算的 Tiling 信息，包含分组数、维度等参数

// 使用 GroupNorm 接口实现 Group Normalization
// 若数据类型T为float且允许修改inputX，可设置isReuseSource = true复用inputX内存空间以节省内存
AscendC::GroupNorm<T, isReuseSource>(
    output,           // 输出：归一化并缩放平移后的结果
    outputMean,       // 输出：每组的均值
    outputVariance,   // 输出：每组的方差
    inputX,           // 输入：原始特征图
    gamma,            // 输入：缩放参数 γ
    beta,             // 输入：偏置参数 β
    epsilon,          // 输入：防止除零的系数 ε
    tiling            // 输入：Tiling 调度信息
);
```

