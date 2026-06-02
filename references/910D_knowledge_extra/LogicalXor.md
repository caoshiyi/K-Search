# LogicalXor<a name="ZH-CN_TOPIC_0000002523304748"></a>

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

按元素进行逻辑异或操作。当输入的数据类型不是bool时，零被视为False，非零数据被视为True。

<!-- img2text -->
$$
out_i = x1_i \oplus x2_i =
\begin{cases}
1, & x1_i \neq x2_i \\
0, & x1_i = x2_i
\end{cases}
$$

## 函数原型<a name="section620mcpsimp"></a>

```
template <const LogicalXorConfig& config = DEFAULT_LOGICAL_XOR_CONFIG, typename T, typename U>
__aicore__ inline void LogicalXor(const LocalTensor<T>& dst, const LocalTensor<U>& src0, const LocalTensor<U>& src1, const uint32_t count)
```

## 参数说明<a name="section91811022141317"></a>

**表 1**  模板参数说明

<a name="table4835205712588"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001429830437_row118356578583"><th class="cellrowborder" valign="top" width="13.65%" id="mcps1.2.3.1.1"><p id="zh-cn_topic_0000001429830437_p48354572582"><a name="zh-cn_topic_0000001429830437_p48354572582"></a><a name="zh-cn_topic_0000001429830437_p48354572582"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="86.35000000000001%" id="mcps1.2.3.1.2"><p id="zh-cn_topic_0000001429830437_p583535795817"><a name="zh-cn_topic_0000001429830437_p583535795817"></a><a name="zh-cn_topic_0000001429830437_p583535795817"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001429830437_row1835857145817"><td class="cellrowborder" valign="top" width="13.65%" headers="mcps1.2.3.1.1 "><p id="p1111714411513"><a name="p1111714411513"></a><a name="p1111714411513"></a>config</p>
</td>
<td class="cellrowborder" valign="top" width="86.35000000000001%" headers="mcps1.2.3.1.2 "><p id="p81171141152"><a name="p81171141152"></a><a name="p81171141152"></a>LogicalXor算法的相关配置。此参数可选配，LogicalXorConfig类型，具体定义如下方代码所示，其中参数的含义是：</p>
<p id="p12990958103015"><a name="p12990958103015"></a><a name="p12990958103015"></a>isReuseSource：该参数预留，传入默认值false即可。</p>
</td>
</tr>
<tr id="row12852153084219"><td class="cellrowborder" valign="top" width="13.65%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001538537601_p47551198266"><a name="zh-cn_topic_0000001538537601_p47551198266"></a><a name="zh-cn_topic_0000001538537601_p47551198266"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="86.35000000000001%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001538537601_p125969172719"><a name="zh-cn_topic_0000001538537601_p125969172719"></a><a name="zh-cn_topic_0000001538537601_p125969172719"></a>目的操作数的数据类型。</p>
<p id="p382544110205"><a name="p382544110205"></a><a name="p382544110205"></a><span id="ph1168842372812"><a name="ph1168842372812"></a><a name="ph1168842372812"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：bool。</p>
</td>
</tr>
<tr id="row118405467348"><td class="cellrowborder" valign="top" width="13.65%" headers="mcps1.2.3.1.1 "><p id="p11463077177"><a name="p11463077177"></a><a name="p11463077177"></a>U</p>
</td>
<td class="cellrowborder" valign="top" width="86.35000000000001%" headers="mcps1.2.3.1.2 "><p id="p193020144171"><a name="p193020144171"></a><a name="p193020144171"></a>源操作数的数据类型。</p>
<p id="p830213147175"><a name="p830213147175"></a><a name="p830213147175"></a><span id="ph1730221471711"><a name="ph1730221471711"></a><a name="ph1730221471711"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：bool、int8_t、uint8_t、int16_t、uint16_t、half、bfloat16_t、int32_t、uint32_t、float、int64_t、uint64_t。</p>
</td>
</tr>
</tbody>
</table>

```
struct LogicalXorConfig {
    bool isReuseSource;
};
```

**表 2**  参数说明

<a name="table1549711469155"></a>
<table><thead align="left"><tr id="row12534194619150"><th class="cellrowborder" valign="top" width="14.510000000000002%" id="mcps1.2.4.1.1"><p id="p115341446121510"><a name="p115341446121510"></a><a name="p115341446121510"></a><strong id="b125344463152"><a name="b125344463152"></a><a name="b125344463152"></a>参数名称</strong></p>
</th>
<th class="cellrowborder" valign="top" width="9.49%" id="mcps1.2.4.1.2"><p id="p8534164621511"><a name="p8534164621511"></a><a name="p8534164621511"></a><strong id="b85341463155"><a name="b85341463155"></a><a name="b85341463155"></a>类型</strong></p>
</th>
<th class="cellrowborder" valign="top" width="76%" id="mcps1.2.4.1.3"><p id="p6534046101518"><a name="p6534046101518"></a><a name="p6534046101518"></a><strong id="b105341546101519"><a name="b105341546101519"></a><a name="b105341546101519"></a>说明</strong></p>
</th>
</tr>
</thead>
<tbody><tr id="row1253413467153"><td class="cellrowborder" valign="top" width="14.510000000000002%" headers="mcps1.2.4.1.1 "><p id="p1534204617157"><a name="p1534204617157"></a><a name="p1534204617157"></a>dst</p>
</td>
<td class="cellrowborder" valign="top" width="9.49%" headers="mcps1.2.4.1.2 "><p id="p3534104620153"><a name="p3534104620153"></a><a name="p3534104620153"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="76%" headers="mcps1.2.4.1.3 "><p id="p19206175051520"><a name="p19206175051520"></a><a name="p19206175051520"></a>目的操作数。</p>
<p id="p5945720195112"><a name="p5945720195112"></a><a name="p5945720195112"></a><span id="zh-cn_topic_0000002523303824_ph173308471594"><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><span id="zh-cn_topic_0000002523303824_ph9902231466"><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816"><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
</td>
</tr>
<tr id="row1880217311431"><td class="cellrowborder" valign="top" width="14.510000000000002%" headers="mcps1.2.4.1.1 "><p id="p10802203313377"><a name="p10802203313377"></a><a name="p10802203313377"></a>src0、src1</p>
</td>
<td class="cellrowborder" valign="top" width="9.49%" headers="mcps1.2.4.1.2 "><p id="p9802143314372"><a name="p9802143314372"></a><a name="p9802143314372"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="76%" headers="mcps1.2.4.1.3 "><p id="p3707144233915"><a name="p3707144233915"></a><a name="p3707144233915"></a>源操作数。</p>
<p id="p1696891819580"><a name="p1696891819580"></a><a name="p1696891819580"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_1"><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_1"><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_1"><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
</td>
</tr>
<tr id="row29684505218"><td class="cellrowborder" valign="top" width="14.510000000000002%" headers="mcps1.2.4.1.1 "><p id="p182005174526"><a name="p182005174526"></a><a name="p182005174526"></a>count</p>
</td>
<td class="cellrowborder" valign="top" width="9.49%" headers="mcps1.2.4.1.2 "><p id="p162001617105213"><a name="p162001617105213"></a><a name="p162001617105213"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="76%" headers="mcps1.2.4.1.3 "><p id="p6704183616254"><a name="p6704183616254"></a><a name="p6704183616254"></a>参与计算的元素个数。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section38228281712"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

-   **不支持源操作数与目的操作数地址重叠。**
-   操作数地址偏移对齐要求请参见[通用说明和约束](通用说明和约束.md)。

## 调用示例<a name="section1258204115715"></a>

```
AscendC::LocalTensor<uint32_t> src0, src1;
AscendC::LocalTensor<bool> dst; 
uint32_t count = 512;
AscendC::LogicalXor(dst, src0, src1, count);
```

结果示例如下：

```
输入数据（src0）:
[-4, 4, 1, -8, ... 3]
输入数据（src1）:
[-1, 3, 1, 5, ... -3]
输出数据（dst）:
[False, False, False, False, ... False]
```

