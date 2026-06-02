# SetKernelMode<a name="ZH-CN_TOPIC_0000002523304834"></a>

## 产品支持情况<a name="section1550532418810"></a>

<a name="table38301303189"></a>
<table><thead align="left"><tr id="row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p1883113061818"><a name="p1883113061818"></a><a name="p1883113061818"></a><span id="ph20833205312295"><a name="ph20833205312295"></a><a name="ph20833205312295"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p783113012187"><a name="p783113012187"></a><a name="p783113012187"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row1272474920205"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p17301775812"><a name="p17301775812"></a><a name="p17301775812"></a><span id="ph2272194216543"><a name="ph2272194216543"></a><a name="ph2272194216543"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p37256491200"><a name="p37256491200"></a><a name="p37256491200"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="zh-cn_topic_0000001963639310_zh-cn_topic_0000001656094169_section259105813316"></a>

针对[分离模式](基本架构.md#section1574769433)，CPU调测时，设置内核模式为单AIV模式，单AIC模式或者MIX模式，以分别支持单AIV矢量算子，单AIC矩阵算子，MIX混合算子的CPU调试。不调用该接口的情况下，默认为MIX模式。为保证算子代码在多个硬件平台兼容，[耦合模式](基本架构.md#section1574769433)下也可以调用，该场景下接口不会生效，不影响正常调试。

## 函数原型<a name="zh-cn_topic_0000001963639310_zh-cn_topic_0000001656094169_section2067518173415"></a>

```
void SetKernelMode(KernelMode mode)
```

## 参数说明<a name="zh-cn_topic_0000001963639310_zh-cn_topic_0000001656094169_section158061867342"></a>

<a name="zh-cn_topic_0000001963639310_zh-cn_topic_0000001656094169_zh-cn_topic_0235751031_table33761356"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001963639310_zh-cn_topic_0000001656094169_zh-cn_topic_0235751031_row27598891"><th class="cellrowborder" valign="top" width="16.49%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001963639310_zh-cn_topic_0000001656094169_zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0000001963639310_zh-cn_topic_0000001656094169_zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0000001963639310_zh-cn_topic_0000001656094169_zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="11.93%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001963639310_zh-cn_topic_0000001656094169_zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0000001963639310_zh-cn_topic_0000001656094169_zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0000001963639310_zh-cn_topic_0000001656094169_zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="71.58%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001963639310_zh-cn_topic_0000001656094169_zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0000001963639310_zh-cn_topic_0000001656094169_zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0000001963639310_zh-cn_topic_0000001656094169_zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001963639310_zh-cn_topic_0000001656094169_row42461942101815"><td class="cellrowborder" valign="top" width="16.49%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001963639310_zh-cn_topic_0000001656094169_p45208478318"><a name="zh-cn_topic_0000001963639310_zh-cn_topic_0000001656094169_p45208478318"></a><a name="zh-cn_topic_0000001963639310_zh-cn_topic_0000001656094169_p45208478318"></a>mode</p>
</td>
<td class="cellrowborder" valign="top" width="11.93%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001963639310_zh-cn_topic_0000001656094169_p135196472314"><a name="zh-cn_topic_0000001963639310_zh-cn_topic_0000001656094169_p135196472314"></a><a name="zh-cn_topic_0000001963639310_zh-cn_topic_0000001656094169_p135196472314"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.58%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001963639310_zh-cn_topic_0000001656094169_p11518154714314"><a name="zh-cn_topic_0000001963639310_zh-cn_topic_0000001656094169_p11518154714314"></a><a name="zh-cn_topic_0000001963639310_zh-cn_topic_0000001656094169_p11518154714314"></a>内核模式，针对AIC，AIV，MIX算子的CPU调试，参数取值分别为AIC_MODE，AIV_MODE，MIX_MODE。</p>
<a name="zh-cn_topic_0000001963639310_zh-cn_topic_0000001656094169_screen15712112124613"></a><a name="zh-cn_topic_0000001963639310_zh-cn_topic_0000001656094169_screen15712112124613"></a><pre class="screen" codetype="Cpp" id="zh-cn_topic_0000001963639310_zh-cn_topic_0000001656094169_screen15712112124613">enum class KernelMode {
    MIX_MODE = 0,
    AIC_MODE,
    AIV_MODE
};</pre>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0000001963639310_zh-cn_topic_0000001656094169_section640mcpsimp"></a>

无

## 调用示例<a name="zh-cn_topic_0000001963639310_zh-cn_topic_0000001656094169_section82241477610"></a>

```
int32_t main(int32_t argc, char* argv[])
{
    ...
#ifdef ASCENDC_CPU_DEBUG
    ...
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(add_custom, numBlocks, x, y, z); // use this macro for cpu debug
    ...
    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)y);
    AscendC::GmFree((void *)z);
#else
    ...
#endif
    return 0;
}
```

