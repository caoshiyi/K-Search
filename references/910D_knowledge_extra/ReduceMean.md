# ReduceMean<a name="ZH-CN_TOPIC_0000002523303780"></a>

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

对一个多维向量按照指定的维度求平均值。

定义指定计算的维度（Reduce轴）为R轴，非指定维度（Normal轴）为A轴。如下图所示，对shape为\(2, 3\)的二维矩阵进行运算，对第一维计算数据求平均值，输出结果为\[2.5, 3.5, 4.5\]；对第二维计算数据求平均值，输出结果为\[2, 5\]。

**图 1**  ReduceMean按第一个维度计算示例<a name="fig9387114683813"></a>  
<!-- img2text -->
```
A轴（Normal轴）

                     按第一个维度（R轴）求均值

R轴（Reduce轴）   ┌─────┬─────┬─────┐      ─────────────────────────→      ┌─────┬─────┬─────┐
                 │  1  │  2  │  3  │                                     │ 2.5 │ 3.5 │ 4.5 │
                 ├─────┼─────┼─────┤                                     └─────┴─────┴─────┘
                 │  4  │  5  │  6  │
                 └─────┴─────┴─────┘
```

**图 2**  ReduceMean最后一个维度计算示例<a name="fig169300246416"></a>  
<!-- img2text -->
```text
R轴（Reduce轴）

A轴（Normal轴）              按最后一个维度（R轴）求均值

                           ┌──────────────────────────→
┌─────┬─────┬─────┐         │
│  1  │  2  │  3  │         │                      ┌─────┐
├─────┼─────┼─────┤         │                      │  2  │
│  4  │  5  │  6  │         │                      ├─────┤
└─────┴─────┴─────┘         │                      │  5  │
                            │                      └─────┘
                            └──────────────────────→
```

## 函数原型<a name="section620mcpsimp"></a>

-   通过sharedTmpBuffer入参传入临时空间

    ```
    template <class T, class pattern, bool isReuseSource = false>
    __aicore__ inline void ReduceMean(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t srcShape[], bool srcInnerPad)
    ```

-   接口框架申请临时空间

    ```
    template <class T, class pattern, bool isReuseSource = false>
    __aicore__ inline void ReduceMean(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor, const uint32_t srcShape[], bool srcInnerPad)
    ```

由于该接口的内部实现中涉及复杂的数学计算，需要额外的临时空间来存储计算过程中的中间变量。临时空间支持开发者**通过sharedTmpBuffer入参传入**和**接口框架申请**两种方式。

-   通过sharedTmpBuffer入参传入，使用该tensor作为临时空间进行处理，接口框架不再申请。该方式开发者可以自行管理sharedTmpBuffer内存空间，并在接口调用完成后，复用该部分内存，内存不会反复申请释放，灵活性较高，内存利用率也较高。
-   接口框架申请临时空间，开发者无需申请，但是需要预留临时空间的大小。

通过sharedTmpBuffer传入的情况，开发者需要为tensor申请空间；接口框架申请的方式，开发者需要预留临时空间。临时空间大小BufferSize的获取方式如下：通过[GetReduceMeanMaxMinTmpSize](GetReduceMeanMaxMinTmpSize.md)中提供的接口获取需要预留空间范围的大小。

## 参数说明<a name="section622mcpsimp"></a>

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
<p id="p1860114143617"><a name="p1860114143617"></a><a name="p1860114143617"></a><span id="ph1928231818522"><a name="ph1928231818522"></a><a name="ph1928231818522"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：float。</p>
</td>
</tr>
<tr id="row0878818204013"><td class="cellrowborder" valign="top" width="19.39%" headers="mcps1.2.3.1.1 "><p id="p1587851844012"><a name="p1587851844012"></a><a name="p1587851844012"></a>pattern</p>
</td>
<td class="cellrowborder" valign="top" width="80.61%" headers="mcps1.2.3.1.2 "><p id="p16565454161520"><a name="p16565454161520"></a><a name="p16565454161520"></a>用于指定ReduceMean计算轴，包括Reduce轴和Normal轴。pattern由与向量维度数量相同的A、R字母组合形成，字母A表示Normal轴，R表示Reduce轴。例如，AR表示对二维向量进行ReduceMean计算：第一维是Normal轴，第二维是Reduce轴，即对第二维数据计算求平均值。</p>
<p id="p832614503206"><a name="p832614503206"></a><a name="p832614503206"></a>pattern是定义在AscendC::Pattern::Reduce命名空间下的结构体，其成员变量用户无需关注。</p>
<p id="p1989522914112"><a name="p1989522914112"></a><a name="p1989522914112"></a>pattern当前只支持取值为AR和RA，当前用户需要显式指定pattern为AscendC::Pattern::Reduce::AR或者AscendC::Pattern::Reduce::RA。</p>
</td>
</tr>
<tr id="row9756719122620"><td class="cellrowborder" valign="top" width="19.39%" headers="mcps1.2.3.1.1 "><p id="p1682112447268"><a name="p1682112447268"></a><a name="p1682112447268"></a>isReuseSource</p>
</td>
<td class="cellrowborder" valign="top" width="80.61%" headers="mcps1.2.3.1.2 "><p id="p2322650204211"><a name="p2322650204211"></a><a name="p2322650204211"></a>是否允许修改源操作数，默认值为false。如果开发者允许源操作数被改写，可以使能该参数，使能后能够节省部分内存空间。</p>
<p id="p183225504424"><a name="p183225504424"></a><a name="p183225504424"></a>设置为<strong id="b732214507424"><a name="b732214507424"></a><a name="b732214507424"></a>true</strong>，则本接口内部计算时<strong id="b5322165017429"><a name="b5322165017429"></a><a name="b5322165017429"></a>复用</strong>src的内存空间，节省内存空间；设置为<strong id="b2322350134218"><a name="b2322350134218"></a><a name="b2322350134218"></a>false</strong>，则本接口内部计算时<strong id="b6322175013421"><a name="b6322175013421"></a><a name="b6322175013421"></a>不复用</strong>src的内存空间。</p>
<p id="p62891018544"><a name="p62891018544"></a><a name="p62891018544"></a>isReuseSource的使用样例请参考<a href="更多样例-104.md#section639165323915">更多样例</a>。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  接口参数说明

<a name="table21393193468"></a>
<table><thead align="left"><tr id="row51398190469"><th class="cellrowborder" valign="top" width="17.380000000000003%" id="mcps1.2.4.1.1"><p id="p1713971915466"><a name="p1713971915466"></a><a name="p1713971915466"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="10.36%" id="mcps1.2.4.1.2"><p id="p31391219124610"><a name="p31391219124610"></a><a name="p31391219124610"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="72.26%" id="mcps1.2.4.1.3"><p id="p11139121912466"><a name="p11139121912466"></a><a name="p11139121912466"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row41391919174610"><td class="cellrowborder" valign="top" width="17.380000000000003%" headers="mcps1.2.4.1.1 "><p id="p15139819124615"><a name="p15139819124615"></a><a name="p15139819124615"></a>dstTensor</p>
</td>
<td class="cellrowborder" valign="top" width="10.36%" headers="mcps1.2.4.1.2 "><p id="p413951944618"><a name="p413951944618"></a><a name="p413951944618"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="72.26%" headers="mcps1.2.4.1.3 "><p id="p2013916190465"><a name="p2013916190465"></a><a name="p2013916190465"></a>目的操作数。</p>
<p id="p18139141912462"><a name="p18139141912462"></a><a name="p18139141912462"></a><span id="zh-cn_topic_0000002523303824_ph173308471594"><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><span id="zh-cn_topic_0000002523303824_ph9902231466"><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816"><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
</td>
</tr>
<tr id="row17140151904610"><td class="cellrowborder" valign="top" width="17.380000000000003%" headers="mcps1.2.4.1.1 "><p id="p314021964614"><a name="p314021964614"></a><a name="p314021964614"></a>srcTensor</p>
</td>
<td class="cellrowborder" valign="top" width="10.36%" headers="mcps1.2.4.1.2 "><p id="p171409196463"><a name="p171409196463"></a><a name="p171409196463"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.26%" headers="mcps1.2.4.1.3 "><p id="p9140121954618"><a name="p9140121954618"></a><a name="p9140121954618"></a>源操作数。</p>
<p id="p18140819144613"><a name="p18140819144613"></a><a name="p18140819144613"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_1"><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_1"><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_1"><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p15221114412247"><a name="p15221114412247"></a><a name="p15221114412247"></a>源操作数的数据类型需要与目的操作数保持一致。</p>
</td>
</tr>
<tr id="row16857119162714"><td class="cellrowborder" valign="top" width="17.380000000000003%" headers="mcps1.2.4.1.1 "><p id="p1313415271911"><a name="p1313415271911"></a><a name="p1313415271911"></a>sharedTmpBuffer</p>
</td>
<td class="cellrowborder" valign="top" width="10.36%" headers="mcps1.2.4.1.2 "><p id="p5133352201914"><a name="p5133352201914"></a><a name="p5133352201914"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.26%" headers="mcps1.2.4.1.3 "><p id="p0400131017545"><a name="p0400131017545"></a><a name="p0400131017545"></a>临时缓存。</p>
<p id="p11947511105415"><a name="p11947511105415"></a><a name="p11947511105415"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_2"><a name="zh-cn_topic_0000002523303824_ph173308471594_2"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_2"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_2"><a name="zh-cn_topic_0000002523303824_ph9902231466_2"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_2"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_2"><a name="zh-cn_topic_0000002523303824_ph1782115034816_2"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_2"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p104071111204211"><a name="p104071111204211"></a><a name="p104071111204211"></a>用于ReduceMean内部复杂计算时存储中间变量，由开发者提供。</p>
<p id="p5881016172817"><a name="p5881016172817"></a><a name="p5881016172817"></a>临时空间大小BufferSize的获取方式请参考<a href="GetReduceMeanMaxMinTmpSize.md">GetReduceMeanMaxMinTmpSize</a>。</p>
</td>
</tr>
<tr id="row161404199468"><td class="cellrowborder" valign="top" width="17.380000000000003%" headers="mcps1.2.4.1.1 "><p id="p6140219194617"><a name="p6140219194617"></a><a name="p6140219194617"></a>srcShape</p>
</td>
<td class="cellrowborder" valign="top" width="10.36%" headers="mcps1.2.4.1.2 "><p id="p191401119154613"><a name="p191401119154613"></a><a name="p191401119154613"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.26%" headers="mcps1.2.4.1.3 "><p id="p8140019184613"><a name="p8140019184613"></a><a name="p8140019184613"></a>uint32_t类型的数组，表示源操作数的shape信息。该shape的维度必须和模板参数pattern的维度一致，例如，pattern为AR，该shape维度只能是二维。</p>
<p id="p58304922610"><a name="p58304922610"></a><a name="p58304922610"></a><span id="ph684144932616"><a name="ph684144932616"></a><a name="ph684144932616"></a>Ascend 950PR/Ascend 950DT</span>，当前只支持二维shape。</p>
</td>
</tr>
<tr id="row13140141919462"><td class="cellrowborder" valign="top" width="17.380000000000003%" headers="mcps1.2.4.1.1 "><p id="p214031944610"><a name="p214031944610"></a><a name="p214031944610"></a>srcInnerPad</p>
</td>
<td class="cellrowborder" valign="top" width="10.36%" headers="mcps1.2.4.1.2 "><p id="p18140919114610"><a name="p18140919114610"></a><a name="p18140919114610"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.26%" headers="mcps1.2.4.1.3 "><p id="p814011918468"><a name="p814011918468"></a><a name="p814011918468"></a>表示实际需要计算的最内层轴数据是否32Bytes对齐。</p>
<p id="p1019111742714"><a name="p1019111742714"></a><a name="p1019111742714"></a><span id="ph3191117162716"><a name="ph3191117162716"></a><a name="ph3191117162716"></a>Ascend 950PR/Ascend 950DT</span>，当前只支持true。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section38228281712"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

-   操作数地址对齐要求请参见[通用地址对齐约束](通用说明和约束.md#section796754519912)。

-   **不支持源操作数与目的操作数地址重叠。**
-   不支持sharedTmpBuffer与源操作数和目的操作数地址重叠。
-   内部算法不处理累加计算时的数据溢出，溢出场景不保证接口精度。

## 调用示例<a name="section642mcpsimp"></a>

```
AscendC::LocalTensor<float> dstLocal = outQueue.AllocTensor<float>();
AscendC::LocalTensor<float> srcLocal = inQueue.DeQue<float>();
AscendC::LocalTensor<uint8_t> tmp = tbuf.Get<uint8_t>();
uint32_t shape[] = { 2, 8 };
constexpr bool isReuse = true;
AscendC::ReduceMean<T, AscendC::Pattern::Reduce::AR, isReuse>(dstLocal, srcLocal, tmp, shape, true);
```

结果示例如下：

```
输入输出的数据类型为float
输入数据(src): 
[[ 0.0 4.0 2.0 0.0 -1.0 2.0 -1.0 7.0],
 [ 0.0 1.0 -9.0 2.0 2.0 2.0 8.0 3.0]]
输入pattern：AR
输入shape：(2,8)
输出数据(dst): [1.625 1.125]
```

