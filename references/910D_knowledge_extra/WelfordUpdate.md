# WelfordUpdate<a name="ZH-CN_TOPIC_0000002554343697"></a>

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

## 功能说明<a name="section618mcpsimp"></a>

Welford是一种在线计算均值和方差的方法。一方面，它可以在不存储所有样本的情况下，逐步计算所有样本的均值和方差，更适合处理海量数据；另一方面，它只需要对数据进行一次遍历，能减少访存次数，提高计算性能。本接口为Welford算法的前处理。

LayerNorm算法中Reduce轴较大的场景，可以通过切分Reduce轴，联合使用本接口与[WelfordFinalize](WelfordFinalize.md)，实现等效计算LayerNorm。

如下图所示，切分数据的Reduce轴，假设切分后每块数据的形状为\[1, k\]，每块数据标号为1，2，3，…，n。

**图 1**  Reduce轴切分示意图<a name="fig14767612174912"></a>  
<!-- img2text -->
```
                         Reduce轴

┌────────────────┬────────────────┬────────────────┬────────────────┐
│       1        │       2        │       …        │       n        │
└────────────────┴────────────────┴────────────────┴────────────────┘
```

本接口的计算公式如下。进行上述的数据切分后，分n次调用本接口，切分后的每块数据均完成如下公式的计算。

<!-- img2text -->
```
            x_i - Meant_{i-1}
Meant_i = Meant_{i-1} + ─────────────
                   i
```

<!-- img2text -->
```text
M_i = M_(i-1) + (x_i - Meant_(i-1))(x_i - Meant_i)
```

上式中，x<sub>i</sub>、Meant<sub>i</sub>、M<sub>i</sub>的形状均为\[1, k\]，x<sub>i</sub>表示切分后的第i块数据，Meant<sub>i</sub>表示第i次调用本接口得到的前i块数据的均值，M<sub>i</sub>表示第i次调用本接口得到的前i块数据的方差中间结果（即为求方差而保存的中间计算结果，本节后续内容中写作方差中间结果）。其中，第一次调用本接口，即i=1时，公式中的Meant<sub>0</sub>和M<sub>0</sub>由用户定义为形状\[1, k\]、取值全0的数据。

Meant<sub>n</sub>的计算过程示意如下图，调用n次本接口后，得到形状为\[1, k\]的Meant<sub>n</sub>和M<sub>n</sub>，Meant<sub>n</sub>和M<sub>n</sub>用于后续[WelfordFinalize](WelfordFinalize.md)接口的计算。

**图 2**  均值Meant<sub>n</sub>计算过程示意图<a name="fig1241371913223"></a>  
<!-- img2text -->
```text
Xi                                 Meant_i

          ┌───────────────┐                    ┌───────────────┐
          │      x1       │ ─────────────────→ │    Meant1     │
          └───────────────┘                    └───────────────┘

          ┌───────────────┐                    ┌───────────────┐
          │      x2       │ ─────────────────→ │    Meant2     │
          └───────────────┘                    └───────────────┘

                    ...                                  ...

          ┌───────────────┐                    ┌───────────────┐
          │     x(n-1)    │ ─────────────────→ │   Meant(n-1)  │
          └───────────────┘                    └───────────────┘
                                                          │
                                                          ↓
          ┌───────────────┐                    ┌───────────────┐
          │      xn       │ ─────────────────→ │    Meantn     │
          └───────────────┘                    └───────────────┘
                                                          ↑
                                                          │
```

说明:
- 左列标题: `X_i`
- 右列标题: `Meant_i`
- 图中右侧 `Meant_i` 为逐次计算结果，`Meant(n-1)` 通过竖向箭头传递到 `Meantn`
- 原图中的下标为数学下标形式，这里用 `x1 / x2 / x(n-1) / xn`、`Meant1 / Meant2 / Meant(n-1) / Meantn` 表示

## 函数原型<a name="section620mcpsimp"></a>

-   通过sharedTmpBuffer入参传入临时空间

    ```
    template <typename T, typename U,bool isReuseSource = false, const WelfordUpdateConfig& config = WFUPDATE_DEFAULT_CFG>
    __aicore__ inline void WelfordUpdate(const LocalTensor<U>& outputMean, const LocalTensor<U>& outputVariance, const LocalTensor<U>& inputMean, const LocalTensor<U>& inputVariance, const LocalTensor<T>& inputX, const LocalTensor<uint8_t>& sharedTmpBuffer, const WelfordUpdateParam& para)
    ```

-   接口框架申请临时空间

    ```
    template <typename T, typename U,bool isReuseSource = false, const WelfordUpdateConfig& config = WFUPDATE_DEFAULT_CFG>
    __aicore__ inline void WelfordUpdate(const LocalTensor<U>& outputMean, const LocalTensor<U>& outputVariance, const LocalTensor<U>& inputMean, const LocalTensor<U>& inputVariance, const LocalTensor<T>& inputX, const WelfordUpdateParam& para)
    ```

由于该接口的内部实现中涉及复杂的计算，需要额外的临时空间来存储计算过程中的中间变量。临时空间支持**接口框架申请**和开发者**通过sharedTmpBuffer入参传入**两种方式。

-   接口框架申请临时空间，开发者无需申请，但是需要预留临时空间的大小。

-   通过sharedTmpBuffer入参传入，使用该tensor作为临时空间进行处理，接口框架不再申请。该方式开发者可以自行管理sharedTmpBuffer内存空间，并在接口调用完成后，复用该部分内存，内存不会反复申请释放，灵活性较高，内存利用率也较高。

接口框架申请的方式，开发者需要预留临时空间；通过sharedTmpBuffer传入的情况，开发者需要为tensor申请空间。临时空间大小BufferSize的获取方式如下：通过[WelfordUpdate Tiling](WelfordUpdate-Tiling.md)中提供的GetWelfordUpdateMaxMinTmpSize接口获取所需最大和最小临时空间大小，最小空间可以保证功能正确，最大空间用于提升性能。

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table575571914269"></a>
<table><thead align="left"><tr id="row18755131942614"><th class="cellrowborder" valign="top" width="19.05%" id="mcps1.2.3.1.1"><p id="p675519193268"><a name="p675519193268"></a><a name="p675519193268"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="80.95%" id="mcps1.2.3.1.2"><p id="p375511918267"><a name="p375511918267"></a><a name="p375511918267"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row14755141911264"><td class="cellrowborder" valign="top" width="19.05%" headers="mcps1.2.3.1.1 "><p id="p47551198266"><a name="p47551198266"></a><a name="p47551198266"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="80.95%" headers="mcps1.2.3.1.2 "><p id="p125969172719"><a name="p125969172719"></a><a name="p125969172719"></a>inputX操作数的数据类型。</p>
<p id="p4243195515810"><a name="p4243195515810"></a><a name="p4243195515810"></a><span id="ph3243145505815"><a name="ph3243145505815"></a><a name="ph3243145505815"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：half、bfloat16_t、float</p>
</td>
</tr>
<tr id="row189317536396"><td class="cellrowborder" valign="top" width="19.05%" headers="mcps1.2.3.1.1 "><p id="p5673141812416"><a name="p5673141812416"></a><a name="p5673141812416"></a>U</p>
</td>
<td class="cellrowborder" valign="top" width="80.95%" headers="mcps1.2.3.1.2 "><p id="p1467371812415"><a name="p1467371812415"></a><a name="p1467371812415"></a>outputMean、outputVariance、inputMean、inputVariance操作数的数据类型。</p>
<p id="p134411232183317"><a name="p134411232183317"></a><a name="p134411232183317"></a><span id="ph1441123293315"><a name="ph1441123293315"></a><a name="ph1441123293315"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：float</p>
</td>
</tr>
<tr id="row9756719122620"><td class="cellrowborder" valign="top" width="19.05%" headers="mcps1.2.3.1.1 "><p id="p1682112447268"><a name="p1682112447268"></a><a name="p1682112447268"></a>isReuseSource</p>
</td>
<td class="cellrowborder" valign="top" width="80.95%" headers="mcps1.2.3.1.2 "><p id="p1275717613718"><a name="p1275717613718"></a><a name="p1275717613718"></a>是否允许修改源操作数，默认值为false。如果开发者允许源操作数被改写，可以使能该参数，使能后能够节省部分内存空间。</p>
<p id="p175786163713"><a name="p175786163713"></a><a name="p175786163713"></a>设置为<strong id="b575706193714"><a name="b575706193714"></a><a name="b575706193714"></a>true</strong>，则本接口内部计算时<strong id="b147578614378"><a name="b147578614378"></a><a name="b147578614378"></a>复用</strong>inputX的内存空间，节省内存空间；设置为<strong id="b475717616379"><a name="b475717616379"></a><a name="b475717616379"></a>false</strong>，则本接口内部计算时<strong id="b157575653719"><a name="b157575653719"></a><a name="b157575653719"></a>不复用</strong>inputX的内存空间。</p>
<p id="p1719533618369"><a name="p1719533618369"></a><a name="p1719533618369"></a>isReuseSource的使用样例请参考<a href="更多样例-104.md#section639165323915">更多样例</a>。</p>
</td>
</tr>
<tr id="row12398185404114"><td class="cellrowborder" valign="top" width="19.05%" headers="mcps1.2.3.1.1 "><p id="p14147121315505"><a name="p14147121315505"></a><a name="p14147121315505"></a>config</p>
</td>
<td class="cellrowborder" valign="top" width="80.95%" headers="mcps1.2.3.1.2 "><p id="p1744211392255"><a name="p1744211392255"></a><a name="p1744211392255"></a>配置非指定计算范围内的目的操作数与源操作数的复用关系。WelfordUpdateConfig类型，定义如下：</p>
<a name="screen16476116195910"></a><a name="screen16476116195910"></a><pre class="screen" codetype="Cpp" id="screen16476116195910">struct WelfordUpdateConfig {
    bool isInplace = false; // 目的操作数是否复用源操作数。
};</pre>
<a name="ul1167113259457"></a><a name="ul1167113259457"></a><ul id="ul1167113259457"><li>isInplace：<a href="#zh-cn_topic_0235751031_table33761356">接口参数</a>para中的abComputeLength参数指定了输入数据内层轴的计算长度，在该指定计算长度之外的输出数据具体为何值，通过本参数设置。本参数表示，在指定计算长度之外的目的操作数是否复用源操作数；若复用，对于指定计算长度之外的输出，直接使用对应位置的源操作数代替输出目的操作数；若不复用，则本接口不会输出计算范围外的目的操作数。<a name="ul11364174572711"></a><a name="ul11364174572711"></a><ul id="ul11364174572711"><li>false：默认值。表示目的操作数不复用源操作数。</li><li>true：表示目的操作数复用源操作数。outputMean复用inputMean，outputVariance复用inputVariance。</li></ul>
</li></ul>
<p id="p76421594583"><a name="p76421594583"></a><a name="p76421594583"></a>配置示例如下：</p>
<a name="screen19241326175913"></a><a name="screen19241326175913"></a><pre class="screen" codetype="Cpp" id="screen19241326175913">constexpr WelfordUpdateConfig WFUPDATE_DEFAULT_CFG = {false};</pre>
<p id="p19442739102517"><a name="p19442739102517"></a><a name="p19442739102517"></a>此参数一般用于配合kernel侧tiling计算的接口使用。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  接口参数说明

<a name="zh-cn_topic_0235751031_table33761356"></a>
<table><thead align="left"><tr id="zh-cn_topic_0235751031_row27598891"><th class="cellrowborder" valign="top" width="17.77%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="9.66%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="72.57000000000001%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row42461942101815"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p112283556514"><a name="p112283556514"></a><a name="p112283556514"></a>outputMean</p>
</td>
<td class="cellrowborder" valign="top" width="9.66%" headers="mcps1.2.4.1.2 "><p id="p186293346150"><a name="p186293346150"></a><a name="p186293346150"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="72.57000000000001%" headers="mcps1.2.4.1.3 "><p id="p242825624218"><a name="p242825624218"></a><a name="p242825624218"></a>均值目的操作数，对应接口公式中的Meant<sub id="sub197461221131316"><a name="sub197461221131316"></a><a name="sub197461221131316"></a>i</sub>。</p>
<p id="p16911647191712"><a name="p16911647191712"></a><a name="p16911647191712"></a><span id="zh-cn_topic_0000002523303824_ph173308471594"><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><span id="zh-cn_topic_0000002523303824_ph9902231466"><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816"><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p9255193274511"><a name="p9255193274511"></a><a name="p9255193274511"></a>shape和源操作数inputMean需要保持一致。</p>
</td>
</tr>
<tr id="row588411475110"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p123181843195412"><a name="p123181843195412"></a><a name="p123181843195412"></a>outputVariance</p>
</td>
<td class="cellrowborder" valign="top" width="9.66%" headers="mcps1.2.4.1.2 "><p id="p1286018337576"><a name="p1286018337576"></a><a name="p1286018337576"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="72.57000000000001%" headers="mcps1.2.4.1.3 "><p id="p1984317542572"><a name="p1984317542572"></a><a name="p1984317542572"></a>方差中间结果目的操作数，对应接口公式中的M<sub id="sub1058983751315"><a name="sub1058983751315"></a><a name="sub1058983751315"></a>i</sub>。</p>
<p id="p1384305465711"><a name="p1384305465711"></a><a name="p1384305465711"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_1"><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_1"><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_1"><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p0843125435713"><a name="p0843125435713"></a><a name="p0843125435713"></a>shape和源操作数inputVariance需要保持一致。</p>
</td>
</tr>
<tr id="row250163914199"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p493856135417"><a name="p493856135417"></a><a name="p493856135417"></a>inputMean</p>
</td>
<td class="cellrowborder" valign="top" width="9.66%" headers="mcps1.2.4.1.2 "><p id="p1662903414157"><a name="p1662903414157"></a><a name="p1662903414157"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.57000000000001%" headers="mcps1.2.4.1.3 "><p id="p082871620172"><a name="p082871620172"></a><a name="p082871620172"></a>均值源操作数，对应接口公式中的Meant<sub id="sub15549814151214"><a name="sub15549814151214"></a><a name="sub15549814151214"></a>i-1</sub>。</p>
<p id="p15450144034510"><a name="p15450144034510"></a><a name="p15450144034510"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_2"><a name="zh-cn_topic_0000002523303824_ph173308471594_2"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_2"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_2"><a name="zh-cn_topic_0000002523303824_ph9902231466_2"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_2"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_2"><a name="zh-cn_topic_0000002523303824_ph1782115034816_2"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_2"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
</td>
</tr>
<tr id="row81221355101812"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p732415106555"><a name="p732415106555"></a><a name="p732415106555"></a>inputVariance</p>
</td>
<td class="cellrowborder" valign="top" width="9.66%" headers="mcps1.2.4.1.2 "><p id="p121236558181"><a name="p121236558181"></a><a name="p121236558181"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.57000000000001%" headers="mcps1.2.4.1.3 "><p id="p4238145325818"><a name="p4238145325818"></a><a name="p4238145325818"></a>方差中间结果源操作数，对应接口公式中的M<sub id="sub15831124171218"><a name="sub15831124171218"></a><a name="sub15831124171218"></a>i-1</sub>。</p>
<p id="p12238353135818"><a name="p12238353135818"></a><a name="p12238353135818"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_3"><a name="zh-cn_topic_0000002523303824_ph173308471594_3"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_3"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_3"><a name="zh-cn_topic_0000002523303824_ph9902231466_3"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_3"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_3"><a name="zh-cn_topic_0000002523303824_ph1782115034816_3"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_3"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
</td>
</tr>
<tr id="row15733201320551"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p19963131619557"><a name="p19963131619557"></a><a name="p19963131619557"></a>inputX</p>
</td>
<td class="cellrowborder" valign="top" width="9.66%" headers="mcps1.2.4.1.2 "><p id="p157331713195513"><a name="p157331713195513"></a><a name="p157331713195513"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.57000000000001%" headers="mcps1.2.4.1.3 "><p id="p192431055125815"><a name="p192431055125815"></a><a name="p192431055125815"></a>源操作数，对应接口公式中的x<sub id="sub365610515133"><a name="sub365610515133"></a><a name="sub365610515133"></a>i</sub>。</p>
<p id="p12243135545816"><a name="p12243135545816"></a><a name="p12243135545816"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_4"><a name="zh-cn_topic_0000002523303824_ph173308471594_4"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_4"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_4"><a name="zh-cn_topic_0000002523303824_ph9902231466_4"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_4"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_4"><a name="zh-cn_topic_0000002523303824_ph1782115034816_4"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_4"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
</td>
</tr>
<tr id="row20749938181910"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p1575033814199"><a name="p1575033814199"></a><a name="p1575033814199"></a>sharedTmpBuffer</p>
</td>
<td class="cellrowborder" valign="top" width="9.66%" headers="mcps1.2.4.1.2 "><p id="p77501738191912"><a name="p77501738191912"></a><a name="p77501738191912"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.57000000000001%" headers="mcps1.2.4.1.3 "><p id="p1323018409190"><a name="p1323018409190"></a><a name="p1323018409190"></a>临时空间。</p>
<p id="p14203184218188"><a name="p14203184218188"></a><a name="p14203184218188"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_5"><a name="zh-cn_topic_0000002523303824_ph173308471594_5"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_5"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_5"><a name="zh-cn_topic_0000002523303824_ph9902231466_5"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_5"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_5"><a name="zh-cn_topic_0000002523303824_ph1782115034816_5"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_5"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p104071111204211"><a name="p104071111204211"></a><a name="p104071111204211"></a>接口内部复杂计算时用于存储中间变量，由开发者提供。</p>
<p id="p2230154015198"><a name="p2230154015198"></a><a name="p2230154015198"></a>临时空间大小BufferSize的获取方式请参考<a href="WelfordUpdate-Tiling.md">WelfordUpdate Tiling</a>。</p>
</td>
</tr>
<tr id="row041413392198"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p054917245550"><a name="p054917245550"></a><a name="p054917245550"></a>para</p>
</td>
<td class="cellrowborder" valign="top" width="9.66%" headers="mcps1.2.4.1.2 "><p id="p263018345154"><a name="p263018345154"></a><a name="p263018345154"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.57000000000001%" headers="mcps1.2.4.1.3 "><p id="p0534135117111"><a name="p0534135117111"></a><a name="p0534135117111"></a>计算所需的参数信息。WelfordUpdateParam类型，定义如下。</p>
<a name="screen1284915182119"></a><a name="screen1284915182119"></a><pre class="screen" codetype="Cpp" id="screen1284915182119">struct WelfordUpdateParam {
    uint32_t rnLength; 
    uint32_t abLength; 
    uint32_t abComputeLength; 
    float nRec;
};</pre>
<a name="ul20925141115211"></a><a name="ul20925141115211"></a><ul id="ul20925141115211"><li>rnLength：预留参数，固定设置为1。</li><li>abLength：Reduce轴拆分的大小。</li><li>abComputeLength：从输入的起始地址开始的Reduce轴实际计算长度。</li><li>nRec：取值为1/i，i为当前调用本接口的累积次数。i的取值范围为[1, n]，n为对输入数据inputX的Reduce轴切分的块数。</li></ul>
<p id="p776215194472"><a name="p776215194472"></a><a name="p776215194472"></a>各目的操作数和源操作数的shape均为[rnLength, abLength]。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

-   接口参数para.rnLength当前只支持取值为1；
-   接口参数para.abLength的取值必须为32/sizeof\(T\)的整数倍；
-   接口参数para.abComputeLength的取值必须大于0。
-   不支持源操作数与目的操作数地址重叠**。**
-   不支持sharedTmpBuffer与源操作数和目的操作数地址重叠。

## 调用示例<a name="section94691236101419"></a>

完整的调用样例可参考[WelfordUpdate样例](https://gitcode.com/cann/asc-devkit/tree/master/examples/03_libraries/03_normalization/welford_update)。

```
// outputMean: 输出更新后的均值 Meant，shape 为 [1, abLength]
// outputVariance: 输出更新后的方差中间结果 Mi，shape 为 [1, abLength]
// inputMean: 上一时刻的均值 Meant-1，作为输入
// inputVariance: 上一时刻的方差中间结果 Mi-1，作为输入
// inputX: 当前时间步的输入数据 xi，shape 为 [1, abLength]
// sharedTmpBuffer: 开发者管理的临时空间，用于内部复杂计算
// para: 包含 Reduce 轴分块信息和归一化系数的参数结构

// 使用 WelfordUpdate 接口执行 Welford 在线算法更新
struct AscendC::WelfordUpdateParam para = { nLength, rLength, abComputeLength, 0.3 };
AscendC::WelfordUpdate<T, U, false, WELFORD_UPDATE_ENABLE_INPLACE_CFG>(
    outputMean,        // 输出：更新后的均值
    outputVariance,    // 输出：更新后的方差中间结果
    inputMean,         // 输入：上一时刻均值
    inputVariance,     // 输入：上一时刻方差中间结果
    inputX,            // 输入：当前输入 xi
    sharedTmpBuffer,   // 输入：临时缓冲区（由开发者提供）
    para               // 输入：Welford 更新参数
);
```

