# Arange<a name="ZH-CN_TOPIC_0000002554344633"></a>

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

给定起始值，等差值和长度，返回一个等差数列。

## 实现原理<a name="section13229175017585"></a>

以float类型，ND格式，firstValue和diffValue输入Scalar为例，描述Arange高阶API内部算法框图，如下图所示。

**图 1**  Arange算法框图<a name="fig574573212104"></a>  
<!-- img2text -->
```text
                               ┌────────────────────────┐
                               │      firstValue,       │
                               │    diffValue, count    │
                               └────────────┬───────────┘
                                            │
                                            ▼
                                      ╱───────────╲
                                     ╱  count > 8  ╲
                                     ╲             ╱
                                      ╲───────────╱
                                       │         │
                                 False │         │ True
                                       │         ▼
                                       │   ┌───────────────┐
                                       │   │   SetValue    │
                                       │   │      [8]      │
                                       │   └───────┬───────┘
                                       │           │
                                       │           ▼
┌───────────┐                          │     ╱────────────╲
│    dst    │◀─────────────────────────┼────╱  count > 64  ╲
└───────────┘                          │    ╲              ╱
                                       │     ╲────────────╱
                                       │        │      │
                                       │   True │      │ False
                                       │        ▼      ▼
                         ┌─────────────┐   ┌───────────────┐
                         │  SetValue   │   │     Adds      │──────────────▶┌───────────┐
                         │   [count]   │   │    [count]    │               │    dst    │
                         └─────────────┘   └───────────────┘               └───────────┘
                                                ▲
                                                │
                                          ┌───────────────┐
                                          │     Adds      │
                                          │      [64]     │
                                          └───────┬───────┘
                                                  │
                                                  ▼
                                          ┌───────────────┐
                                          │     Adds      │
                                          │    [count]    │
                                          └───────┬───────┘
                                                  │
                                                  ▼
                                             ┌───────────┐
                                             │    dst    │
                                             └───────────┘


外层流程边界:
┌──────────────────────────────────────────────────────────────────────────────────────────────┐
│  firstValue, diffValue, count 进入流程                                                     │
│  ├─ 若 count ≤ 8  ：执行 SetValue[count]，输出 dst                                         │
│  ├─ 若 count > 8  ：先执行 SetValue[8]                                                     │
│  │   ├─ 若 count ≤ 64 ：执行 Adds[count]，输出 dst                                         │
│  │   └─ 若 count > 64 ：先执行 Adds[64]，再执行 Adds[count]，输出 dst                      │
└──────────────────────────────────────────────────────────────────────────────────────────────┘


图示:
┌──────────────┐  输入输出Tensor
│              │
└──────────────┘

┌──────────────┐  vector计算
│              │
└──────────────┘

      ╱──────╲
     ╱        ╲   条件判断
     ╲        ╱
      ╲──────╱

────────────▶  数据流向
```

计算过程分为如下几步，均在Vector上进行：

1.  等差数列长度8以内步骤：按照firstValue和diffValue的值，使用SetValue实现等差数列扩充，扩充长度最大为8，如果等差数列长度小于8，算法结束；
2.  等差数列长度8至64的步骤：对第一步中的等差数列结果使用Adds进行扩充，最大循环7次扩充至64，如果等差数列长度小于64，算法结束；
3.  等差数列长度64以上的步骤：对第二步中的等差数列结果使用Adds进行扩充，不断循环直至达到等差数列长度为止。

## 函数原型<a name="section620mcpsimp"></a>

```
template <typename T>
__aicore__ inline void Arange(const LocalTensor<T>& dst, const T firstValue, const T diffValue, const int32_t count)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table729818506422"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001692058420_row11299950204217"><th class="cellrowborder" valign="top" width="19.18%" id="mcps1.2.3.1.1"><p id="zh-cn_topic_0000001692058420_p1029955044218"><a name="zh-cn_topic_0000001692058420_p1029955044218"></a><a name="zh-cn_topic_0000001692058420_p1029955044218"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="80.82000000000001%" id="mcps1.2.3.1.2"><p id="zh-cn_topic_0000001692058420_p1629911506421"><a name="zh-cn_topic_0000001692058420_p1629911506421"></a><a name="zh-cn_topic_0000001692058420_p1629911506421"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001692058420_row12299165018421"><td class="cellrowborder" valign="top" width="19.18%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001692058420_p1329915004219"><a name="zh-cn_topic_0000001692058420_p1329915004219"></a><a name="zh-cn_topic_0000001692058420_p1329915004219"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="80.82000000000001%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001692058420_p8299155010420"><a name="zh-cn_topic_0000001692058420_p8299155010420"></a><a name="zh-cn_topic_0000001692058420_p8299155010420"></a>操作数的数据类型。</p>
<p id="p1011743616262"><a name="p1011743616262"></a><a name="p1011743616262"></a><span id="ph16117636172610"><a name="ph16117636172610"></a><a name="ph16117636172610"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：int16_t、half、int32_t、float、int64_t。</p>
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
<tbody><tr id="row42461942101815"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p410318922218"><a name="p410318922218"></a><a name="p410318922218"></a>dst</p>
</td>
<td class="cellrowborder" valign="top" width="9.69%" headers="mcps1.2.4.1.2 "><p id="p186293346150"><a name="p186293346150"></a><a name="p186293346150"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="72.54%" headers="mcps1.2.4.1.3 "><p id="p9255193274511"><a name="p9255193274511"></a><a name="p9255193274511"></a>目的操作数。dst的大小应大于等于count * sizeof(T)。</p>
<p id="p16703131355116"><a name="p16703131355116"></a><a name="p16703131355116"></a><span id="zh-cn_topic_0000002523303824_ph173308471594"><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><span id="zh-cn_topic_0000002523303824_ph9902231466"><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816"><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
</td>
</tr>
<tr id="row250163914199"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p1762920347151"><a name="p1762920347151"></a><a name="p1762920347151"></a>firstValue</p>
</td>
<td class="cellrowborder" valign="top" width="9.69%" headers="mcps1.2.4.1.2 "><p id="p1662903414157"><a name="p1662903414157"></a><a name="p1662903414157"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.54%" headers="mcps1.2.4.1.3 "><p id="p936684718225"><a name="p936684718225"></a><a name="p936684718225"></a>等差数列的首个元素值。</p>
</td>
</tr>
<tr id="row041413392198"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p4630634141515"><a name="p4630634141515"></a><a name="p4630634141515"></a>diffValue</p>
</td>
<td class="cellrowborder" valign="top" width="9.69%" headers="mcps1.2.4.1.2 "><p id="p263018345154"><a name="p263018345154"></a><a name="p263018345154"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.54%" headers="mcps1.2.4.1.3 "><p id="p132171537112315"><a name="p132171537112315"></a><a name="p132171537112315"></a>等差数列元素之间的差值，应大于等于0。</p>
</td>
</tr>
<tr id="row17371444131520"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p1838644151511"><a name="p1838644151511"></a><a name="p1838644151511"></a>count</p>
</td>
<td class="cellrowborder" valign="top" width="9.69%" headers="mcps1.2.4.1.2 "><p id="p73844410158"><a name="p73844410158"></a><a name="p73844410158"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.54%" headers="mcps1.2.4.1.3 "><p id="p11389449156"><a name="p11389449156"></a><a name="p11389449156"></a>等差数列的长度。count&gt;0。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

当前仅支持ND格式的输入，不支持其他格式。

## 调用示例<a name="section94691236101419"></a>

```
AscendC::LocalTensor<T> dst = outDst.AllocTensor<T>();
AscendC::Arange<T>(dst, static_cast<T>(firstValue_), static_cast<T>(diffValue_), count_);
outDst.EnQue<T>(dst);
```

