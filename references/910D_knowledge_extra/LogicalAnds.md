# LogicalAnds<a name="ZH-CN_TOPIC_0000002554424645"></a>

## 产品支持情况<a name="section1550532418810"></a>

<a name="table1334714391211"></a>
<table><thead align="left"><tr id="row1334743121213"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p834713321216"><a name="p834713321216"></a><a name="p834713321216"></a><span id="ph834783101215"><a name="ph834783101215"></a><a name="ph834783101215"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p2347234127"><a name="p2347234127"></a><a name="p2347234127"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row113472312122"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p234710320128"><a name="p234710320128"></a><a name="p234710320128"></a><span id="ph103471336127"><a name="ph103471336127"></a><a name="ph103471336127"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p4751940181211"><a name="p4751940181211"></a><a name="p4751940181211"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section618mcpsimp"></a>

输入矢量内的每个元素与标量进行与操作。当输入矢量、标量的数据类型不是bool时，零被视为False，非零数据被视为True。接口中矢量与标量的顺序支持标量在前和标量在后两种场景，其中，标量支持配置为LocalTensor的单点元素。计算公式如下，idx表示LocalTensor单点元素的位置系数：

<!-- img2text -->
$$
dst_i =
\begin{cases}
src0_i \land scalar, & \text{标量在后} \\
scalar \land src1_i, & \text{标量在前}
\end{cases}
$$

$$
scalar = scalar[idx]
$$

## 函数原型<a name="section620mcpsimp"></a>

```
template <const LogicalAndsConfig& config = DEFAULT_LOGICAL_ANDS_CONFIG, typename T, typename U, typename S>
__aicore__ inline void LogicalAnds(const LocalTensor<T>& dst, const U& src0, const S& src1, const uint32_t count)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table575571914269"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001538537601_row18755131942614"><th class="cellrowborder" valign="top" width="19.39%" id="mcps1.2.3.1.1"><p id="zh-cn_topic_0000001538537601_p675519193268"><a name="zh-cn_topic_0000001538537601_p675519193268"></a><a name="zh-cn_topic_0000001538537601_p675519193268"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="80.61%" id="mcps1.2.3.1.2"><p id="zh-cn_topic_0000001538537601_p375511918267"><a name="zh-cn_topic_0000001538537601_p375511918267"></a><a name="zh-cn_topic_0000001538537601_p375511918267"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row1727210582147"><td class="cellrowborder" valign="top" width="19.39%" headers="mcps1.2.3.1.1 "><p id="p4346185914149"><a name="p4346185914149"></a><a name="p4346185914149"></a>config</p>
</td>
<td class="cellrowborder" valign="top" width="80.61%" headers="mcps1.2.3.1.2 "><p id="p6346115917149"><a name="p6346115917149"></a><a name="p6346115917149"></a>LogicalAnds算法的相关配置。此参数可选配，LogicalAndsConfig类型，具体定义如下方代码所示，其中参数的含义为：</p>
<p id="p7479129172810"><a name="p7479129172810"></a><a name="p7479129172810"></a>isReuseSource：该参数预留，传入默认值false即可。</p>
<p id="p1547989132819"><a name="p1547989132819"></a><a name="p1547989132819"></a>scalarTensorIndex：当标量为LocalTensor的单点元素时，该参数用于指定标量作为与操作中的左操作数或右操作数，0表示左操作数，1（默认值）表示右操作数。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001538537601_row14755141911264"><td class="cellrowborder" valign="top" width="19.39%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001538537601_p47551198266"><a name="zh-cn_topic_0000001538537601_p47551198266"></a><a name="zh-cn_topic_0000001538537601_p47551198266"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="80.61%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001538537601_p125969172719"><a name="zh-cn_topic_0000001538537601_p125969172719"></a><a name="zh-cn_topic_0000001538537601_p125969172719"></a>目的操作数的数据类型。</p>
<p id="p382544110205"><a name="p382544110205"></a><a name="p382544110205"></a><span id="ph1168842372812"><a name="ph1168842372812"></a><a name="ph1168842372812"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：bool。</p>
</td>
</tr>
<tr id="row15186940142711"><td class="cellrowborder" valign="top" width="19.39%" headers="mcps1.2.3.1.1 "><p id="p11463077177"><a name="p11463077177"></a><a name="p11463077177"></a>U</p>
</td>
<td class="cellrowborder" valign="top" width="80.61%" headers="mcps1.2.3.1.2 "><p id="p1365194415917"><a name="p1365194415917"></a><a name="p1365194415917"></a>LocalTensor类型或标量类型。根据输入参数src0自动推导类型，开发者无需配置该参数，保证src0满足数据类型的约束即可。</p>
<p id="p830213147175"><a name="p830213147175"></a><a name="p830213147175"></a><span id="ph1730221471711"><a name="ph1730221471711"></a><a name="ph1730221471711"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：bool、int8_t、uint8_t、int16_t、uint16_t、half、bfloat16_t、int32_t、uint32_t、float、int64_t、uint64_t。</p>
</td>
</tr>
<tr id="row772410544565"><td class="cellrowborder" valign="top" width="19.39%" headers="mcps1.2.3.1.1 "><p id="p4724554175615"><a name="p4724554175615"></a><a name="p4724554175615"></a>S</p>
</td>
<td class="cellrowborder" valign="top" width="80.61%" headers="mcps1.2.3.1.2 "><p id="p12608191698"><a name="p12608191698"></a><a name="p12608191698"></a>LocalTensor类型或标量类型。根据输入参数src1自动推导类型，开发者无需配置该参数，保证src1满足数据类型的约束即可。</p>
<p id="p12676122155712"><a name="p12676122155712"></a><a name="p12676122155712"></a><span id="ph106761222105713"><a name="ph106761222105713"></a><a name="ph106761222105713"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：bool、int8_t、uint8_t、int16_t、uint16_t、half、bfloat16_t、int32_t、uint32_t、float、int64_t、uint64_t。</p>
</td>
</tr>
</tbody>
</table>

```
struct LogicalAndsConfig {
    bool isReuseSource;
    int8_t scalarTensorIndex;
};
```

**表 2**  接口参数说明

<a name="table148471830151913"></a>
<table><thead align="left"><tr id="row1984733010194"><th class="cellrowborder" valign="top" width="17.380000000000003%" id="mcps1.2.4.1.1"><p id="p2847730181917"><a name="p2847730181917"></a><a name="p2847730181917"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="10.38%" id="mcps1.2.4.1.2"><p id="p58476303197"><a name="p58476303197"></a><a name="p58476303197"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="72.24000000000001%" id="mcps1.2.4.1.3"><p id="p10847203021913"><a name="p10847203021913"></a><a name="p10847203021913"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row98477303196"><td class="cellrowborder" valign="top" width="17.380000000000003%" headers="mcps1.2.4.1.1 "><p id="p15847183018194"><a name="p15847183018194"></a><a name="p15847183018194"></a>dst</p>
</td>
<td class="cellrowborder" valign="top" width="10.38%" headers="mcps1.2.4.1.2 "><p id="p148471930161917"><a name="p148471930161917"></a><a name="p148471930161917"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="72.24000000000001%" headers="mcps1.2.4.1.3 "><p id="p17444349398"><a name="p17444349398"></a><a name="p17444349398"></a>目的操作数。</p>
<p id="p16703131355116"><a name="p16703131355116"></a><a name="p16703131355116"></a><span id="zh-cn_topic_0000002523303824_ph173308471594"><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><span id="zh-cn_topic_0000002523303824_ph9902231466"><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816"><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
</td>
</tr>
<tr id="row11848103091920"><td class="cellrowborder" valign="top" width="17.380000000000003%" headers="mcps1.2.4.1.1 "><p id="p58481330191917"><a name="p58481330191917"></a><a name="p58481330191917"></a>src0、src1</p>
</td>
<td class="cellrowborder" valign="top" width="10.38%" headers="mcps1.2.4.1.2 "><p id="p158485305196"><a name="p158485305196"></a><a name="p158485305196"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.24000000000001%" headers="mcps1.2.4.1.3 "><p id="p3707144233915"><a name="p3707144233915"></a><a name="p3707144233915"></a>源操作数。类型为标量或LocalTensor，类型为LocalTensor时，可以为矢量操作数或标量单点元素，支持的TPosition为VECIN/VECCALC/VECOUT。</p>
</td>
</tr>
<tr id="row216545817417"><td class="cellrowborder" valign="top" width="17.380000000000003%" headers="mcps1.2.4.1.1 "><p id="p1949611581317"><a name="p1949611581317"></a><a name="p1949611581317"></a>count</p>
</td>
<td class="cellrowborder" valign="top" width="10.38%" headers="mcps1.2.4.1.2 "><p id="p174961758436"><a name="p174961758436"></a><a name="p174961758436"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.24000000000001%" headers="mcps1.2.4.1.3 "><p id="p184961858133"><a name="p184961858133"></a><a name="p184961858133"></a>参与计算的元素个数。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section38228281712"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

-   本接口与操作的左操作数及右操作数中必须有一个为矢量，当前不支持左右操作数同时为标量。
-   当传入LocalTensor的单点数据作为标量时，scalarTensorIndex参数需要传入编译期已知的常量，如果传入变量，则需要将该变量声明为constexpr。

-   **不支持源操作数与目的操作数地址重叠。**
-   操作数地址对齐要求请参见[通用地址对齐约束](通用说明和约束.md#section796754519912)。

## 调用示例<a name="section642mcpsimp"></a>

```
AscendC::LocalTensor<bool> dst;
AscendC::LocalTensor<half> src0, src1;
uint32_t count = 512;
// 标量在后
AscendC::LogicalAnds(dst, src0, src1, count); 
// 标量在前
static constexpr AscendC::LogicalAndsConfig config = { false, 0 };
AscendC::LogicalAnds<config>(dst, src0， src1, count); 
```

结果示例如下：

```
// 标量在后
输入数据（src0）: 
[0.4646, 0.2520, 0.3884, 0.0000, 0.2904, 0.0000, 0.5690, 0.2191, 0.7354,
 0.0000, 0.8093, 0.5932, 0.2688, 0.0830, 0.5074, 0.5595, 0.1468, 0.7020,
 ... 0.0238]
输入数据（src1）: 
[3.0]
输出数据（dst）: 
[ True,  True,  True,  False,  True,  False,  True,  True,  True,  False,
  True,  True,  True,  True,  True,  True,  True,  True,  ...  True]
// 标量在前
输入数据（src1）: 
[0.4646, 0.2520, 0.3884, 0.0000, 0.2904, 0.0000, 0.5690, 0.2191, 0.7354,
 0.0000, 0.8093, 0.5932, 0.2688, 0.0830, 0.5074, 0.5595, 0.1468, 0.7020,
 ... 0.0238]
输入数据（src0）: 
[3.0]
输出数据（dst）: 
[ True,  True,  True,  False, True,  False,  True,  True,  True,  False,
  True,  True,  True,  True,  True,  True,  True,  True,  ...  True]
```

