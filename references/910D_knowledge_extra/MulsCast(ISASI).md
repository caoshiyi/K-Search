# MulsCast\(ISASI\)<a name="ZH-CN_TOPIC_0000002554424073"></a>

## 产品支持情况<a name="section1550532418810"></a>

<a name="table38301303189"></a>
<table><thead align="left"><tr id="row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p1883113061818"><a name="p1883113061818"></a><a name="p1883113061818"></a><span id="ph20833205312295"><a name="ph20833205312295"></a><a name="ph20833205312295"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p783113012187"><a name="p783113012187"></a><a name="p783113012187"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row1272474920205"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p12300735171314"><a name="p12300735171314"></a><a name="p12300735171314"></a><span id="ph730011352138"><a name="ph730011352138"></a><a name="ph730011352138"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p37256491200"><a name="p37256491200"></a><a name="p37256491200"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section618mcpsimp"></a>

将矢量源操作数前count个数据与标量相乘再按照CAST\_ROUND模式转换成half类型， 并将计算结果写入dst，此接口支持标量在前和标量在后两种场景。计算公式如下。

<!-- img2text -->
$$dst_i=\mathrm{half}(x_i\times scalar),\ i\in[0,\ count)$$

其中标量输入也支持配置LocalTensor单点元素，计算公式如下，idx表示LocalTensor单点元素的位置系数。

<!-- img2text -->
$$dst_i=\left(src0_i \times src1[idx]\right) \gg shift$$

## 函数原型<a name="section620mcpsimp"></a>

```
template <typename T0 = BinaryDefaultType, typename T1 = BinaryDefaultType, const BinaryConfig &config = DEFAULT_BINARY_CONFIG, typename T2, typename T3, typename T4>
__aicore__ inline void MulsCast(const T2 &dst, const T3 &src0, const T4 &src1, const uint32_t count)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table4835205712588"></a>
<table><thead align="left"><tr id="row118356578583"><th class="cellrowborder" valign="top" width="18.61%" id="mcps1.2.3.1.1"><p id="p48354572582"><a name="p48354572582"></a><a name="p48354572582"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="81.39%" id="mcps1.2.3.1.2"><p id="p583535795817"><a name="p583535795817"></a><a name="p583535795817"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row1835857145817"><td class="cellrowborder" valign="top" width="18.61%" headers="mcps1.2.3.1.1 "><p id="p5835457165816"><a name="p5835457165816"></a><a name="p5835457165816"></a>T0</p>
</td>
<td class="cellrowborder" valign="top" width="81.39%" headers="mcps1.2.3.1.2 "><p id="p168351657155818"><a name="p168351657155818"></a><a name="p168351657155818"></a>目的操作数数据类型。</p>
<p id="p17344122173213"><a name="p17344122173213"></a><a name="p17344122173213"></a><span id="ph117921554193319"><a name="ph117921554193319"></a><a name="ph117921554193319"></a>预留参数，暂未启用，为后续的功能扩展做保留，需要指定时，传入默认值BinaryDefaultType即可。</span></p>
</td>
</tr>
<tr id="row1448571211232"><td class="cellrowborder" valign="top" width="18.61%" headers="mcps1.2.3.1.1 "><p id="p20119515162311"><a name="p20119515162311"></a><a name="p20119515162311"></a>T1</p>
</td>
<td class="cellrowborder" valign="top" width="81.39%" headers="mcps1.2.3.1.2 "><p id="p7484161220233"><a name="p7484161220233"></a><a name="p7484161220233"></a>源操作数数据类型。</p>
<p id="p8484121215238"><a name="p8484121215238"></a><a name="p8484121215238"></a><span id="ph1648417126239"><a name="ph1648417126239"></a><a name="ph1648417126239"></a>预留参数，暂未启用，为后续的功能扩展做保留，需要指定时，传入默认值BinaryDefaultType即可。</span></p>
</td>
</tr>
<tr id="row211319191271"><td class="cellrowborder" valign="top" width="18.61%" headers="mcps1.2.3.1.1 "><p id="p4208135520361"><a name="p4208135520361"></a><a name="p4208135520361"></a>config</p>
</td>
<td class="cellrowborder" valign="top" width="81.39%" headers="mcps1.2.3.1.2 "><p id="p82081255153612"><a name="p82081255153612"></a><a name="p82081255153612"></a>类型为BinaryConfig，当标量为LocalTensor单点元素类型时生效，用于指定单点元素操作数位置。默认值DEFAULT_BINARY_CONFIG，表示右操作数为标量。</p>
<a name="screen420895553620"></a><a name="screen420895553620"></a><pre class="screen" codetype="Cpp" id="screen420895553620">struct BinaryConfig {
    int8_t scalarTensorIndex = 1; // 用于指定标量为LocalTensor单点元素时标量的位置，0表示左操作数，1表示右操作数
};
constexpr BinaryConfig DEFAULT_BINARY_CONFIG = {1};</pre>
</td>
</tr>
<tr id="row9113219132715"><td class="cellrowborder" valign="top" width="18.61%" headers="mcps1.2.3.1.1 "><p id="p1617215407362"><a name="p1617215407362"></a><a name="p1617215407362"></a>T2</p>
</td>
<td class="cellrowborder" valign="top" width="81.39%" headers="mcps1.2.3.1.2 "><p id="p3172440173612"><a name="p3172440173612"></a><a name="p3172440173612"></a>LocalTensor类型，根据输入参数dst自动推导相应的数据类型，开发者无需配置该参数，保证dst满足数据类型的约束即可。</p>
</td>
</tr>
<tr id="row71131019152712"><td class="cellrowborder" valign="top" width="18.61%" headers="mcps1.2.3.1.1 "><p id="p191723408365"><a name="p191723408365"></a><a name="p191723408365"></a>T3</p>
</td>
<td class="cellrowborder" valign="top" width="81.39%" headers="mcps1.2.3.1.2 "><p id="p151721640153610"><a name="p151721640153610"></a><a name="p151721640153610"></a>LocalTensor类型或标量类型，根据输入参数src0自动推导相应的数据类型，开发者无需配置该参数，保证src0满足数据类型的约束即可。</p>
</td>
</tr>
<tr id="row71131919202715"><td class="cellrowborder" valign="top" width="18.61%" headers="mcps1.2.3.1.1 "><p id="p1717216409367"><a name="p1717216409367"></a><a name="p1717216409367"></a>T4</p>
</td>
<td class="cellrowborder" valign="top" width="81.39%" headers="mcps1.2.3.1.2 "><p id="p11721940193620"><a name="p11721940193620"></a><a name="p11721940193620"></a>LocalTensor类型或标量类型，根据输入参数src1自动推导相应的数据类型，开发者无需配置该参数，保证src1满足数据类型的约束即可。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  参数说明

<a name="zh-cn_topic_0235751031_table33761356"></a>
<table><thead align="left"><tr id="zh-cn_topic_0235751031_row27598891"><th class="cellrowborder" valign="top" width="18.54%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="10.05%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="71.41%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row42461942101815"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p284425844311"><a name="p284425844311"></a><a name="p284425844311"></a>dst</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p158449584436"><a name="p158449584436"></a><a name="p158449584436"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p6389114616514"><a name="p6389114616514"></a><a name="p6389114616514"></a>目的操作数。</p>
<p id="p5945720195112"><a name="p5945720195112"></a><a name="p5945720195112"></a><span id="zh-cn_topic_0000002523303824_ph173308471594"><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><span id="zh-cn_topic_0000002523303824_ph9902231466"><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816"><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p2811183544"><a name="p2811183544"></a><a name="p2811183544"></a><span id="ph1479701815419"><a name="ph1479701815419"></a><a name="ph1479701815419"></a>LocalTensor的起始地址需要32字节对齐。</span></p>
<p id="p468305719192"><a name="p468305719192"></a><a name="p468305719192"></a><span id="ph126252025205"><a name="ph126252025205"></a><a name="ph126252025205"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：half</p>
<p id="p84471955113219"><a name="p84471955113219"></a><a name="p84471955113219"></a>不同数据类型对应的精度转换规则见<a href="#table235404962912">表3</a>。</p>
</td>
</tr>
<tr id="row2137145181815"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p943725919217"><a name="p943725919217"></a><a name="p943725919217"></a>src0/src1</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p34371759625"><a name="p34371759625"></a><a name="p34371759625"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p88022033103718"><a name="p88022033103718"></a><a name="p88022033103718"></a>源操作数。</p>
<a name="ul1343573724314"></a><a name="ul1343573724314"></a><ul id="ul1343573724314"><li>类型为LocalTensor时，支持当作矢量操作数或标量单点元素，支持的TPosition为VECIN/VECCALC/VECOUT。<p id="p9802163310371"><a name="p9802163310371"></a><a name="p9802163310371"></a><span id="ph12803633203712"><a name="ph12803633203712"></a><a name="ph12803633203712"></a>LocalTensor的起始地址需要32字节对齐。</span></p>
<p id="p2803143314373"><a name="p2803143314373"></a><a name="p2803143314373"></a><span id="ph28036339376"><a name="ph28036339376"></a><a name="ph28036339376"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：float</p>
</li><li>类型为标量时：<p id="p157482021194415"><a name="p157482021194415"></a><a name="p157482021194415"></a><span id="ph1180353323718"><a name="ph1180353323718"></a><a name="ph1180353323718"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：float</p>
</li></ul>
</td>
</tr>
<tr id="row891912431168"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p195756503168"><a name="p195756503168"></a><a name="p195756503168"></a>count</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p457515071618"><a name="p457515071618"></a><a name="p457515071618"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p044121644612"><a name="p044121644612"></a><a name="p044121644612"></a>参与计算的元素个数。</p>
</td>
</tr>
</tbody>
</table>

**表 3**  精度转换规则

<a name="table235404962912"></a>
<table><thead align="left"><tr id="row935554942920"><th class="cellrowborder" valign="top" width="10.191019101910191%" id="mcps1.2.4.1.1"><p id="p13355144922911"><a name="p13355144922911"></a><a name="p13355144922911"></a>src类型</p>
</th>
<th class="cellrowborder" valign="top" width="9.43094309430943%" id="mcps1.2.4.1.2"><p id="p135514913299"><a name="p135514913299"></a><a name="p135514913299"></a>dst类型</p>
</th>
<th class="cellrowborder" valign="top" width="80.37803780378037%" id="mcps1.2.4.1.3"><p id="p7113121774314"><a name="p7113121774314"></a><a name="p7113121774314"></a>类型转换模式介绍</p>
</th>
</tr>
</thead>
<tbody><tr id="row3355849152915"><td class="cellrowborder" align="left" valign="top" width="10.191019101910191%" headers="mcps1.2.4.1.1 "><p id="p9687163213521"><a name="p9687163213521"></a><a name="p9687163213521"></a>float</p>
</td>
<td class="cellrowborder" align="left" valign="top" width="9.43094309430943%" headers="mcps1.2.4.1.2 "><p id="p113551749202919"><a name="p113551749202919"></a><a name="p113551749202919"></a>half</p>
</td>
<td class="cellrowborder" valign="top" width="80.37803780378037%" headers="mcps1.2.4.1.3 "><p id="p43071952153512"><a name="p43071952153512"></a><a name="p43071952153512"></a>将源操作数按照CAST_ROUND模式取到half所能表示的数，以half格式（溢出默认按照饱和处理）存入dst中。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

-   左操作数及右操作数中，必须有一个为矢量；当前不支持左右操作数同时为标量。
-   本接口传入LocalTensor单点数据作为标量时，idx参数需要传入编译期已知的常量，传入变量时需要声明为constexpr。

## 调用示例<a name="section642mcpsimp"></a>

```
// 标量在后示例
AscendC::MulsCast(dstLocal, src0Local, src1Local[0], 512);

// 标量在前示例
static constexpr AscendC::BinaryConfig config = { 0 };
AscendC::MulsCast<BinaryDefaultType, BinaryDefaultType, config>(dstLocal, src0Local[0], src1Local, 512);
```

结果示例如下：

```
输入数据src0Local：[6 5 11 ... ]
输入数据src1Local：2
输出数据dstLocal：[12 10 22 ... ]
```

