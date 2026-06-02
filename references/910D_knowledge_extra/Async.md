# Async<a name="ZH-CN_TOPIC_0000002554423543"></a>

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

## 功能说明<a name="section618mcpsimp"></a>

Async提供了一个统一的接口，用于在不同模式下（AIC或AIV）执行特定函数，从而避免代码中直接的硬件条件判断（如使用ASCEND\_IS\_AIV或ASCEND\_IS\_AIC）。

## 函数原型<a name="section620mcpsimp"></a>

```
template <EngineType engine, auto funPtr, class... Args>
__aicore__ void Async(Args... args)
```

## 参数说明<a name="section176711403104"></a>

**表 1**  模板参数说明

<a name="table4835205712588"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001429830437_row118356578583"><th class="cellrowborder" valign="top" width="16.28%" id="mcps1.2.3.1.1"><p id="zh-cn_topic_0000001429830437_p48354572582"><a name="zh-cn_topic_0000001429830437_p48354572582"></a><a name="zh-cn_topic_0000001429830437_p48354572582"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="83.72%" id="mcps1.2.3.1.2"><p id="zh-cn_topic_0000001429830437_p583535795817"><a name="zh-cn_topic_0000001429830437_p583535795817"></a><a name="zh-cn_topic_0000001429830437_p583535795817"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row52838432246"><td class="cellrowborder" valign="top" width="16.28%" headers="mcps1.2.3.1.1 "><p id="p4283143162419"><a name="p4283143162419"></a><a name="p4283143162419"></a>engine</p>
</td>
<td class="cellrowborder" valign="top" width="83.72%" headers="mcps1.2.3.1.2 "><p id="p14314255252"><a name="p14314255252"></a><a name="p14314255252"></a>引擎模式，参数取值分别为AIC、AIV。</p>
<a name="screen1431192522519"></a><a name="screen1431192522519"></a><pre class="screen" codetype="Cpp" id="screen1431192522519">enum class EngineType : int32_t {
    AIC = 1, // 仅AIC
    AIV = 2  // 仅AIV
};</pre>
</td>
</tr>
<tr id="row1673605372520"><td class="cellrowborder" valign="top" width="16.28%" headers="mcps1.2.3.1.1 "><p id="p1173716533257"><a name="p1173716533257"></a><a name="p1173716533257"></a>funPtr</p>
</td>
<td class="cellrowborder" valign="top" width="83.72%" headers="mcps1.2.3.1.2 "><p id="p3737105310258"><a name="p3737105310258"></a><a name="p3737105310258"></a>函数指针，指定要执行的函数，函数签名和参数类型由class... Args决定。</p>
</td>
</tr>
<tr id="row14541046131"><td class="cellrowborder" valign="top" width="16.28%" headers="mcps1.2.3.1.1 "><p id="p745418461632"><a name="p745418461632"></a><a name="p745418461632"></a>class... Args</p>
</td>
<td class="cellrowborder" valign="top" width="83.72%" headers="mcps1.2.3.1.2 "><p id="p16454746238"><a name="p16454746238"></a><a name="p16454746238"></a>可变参数模板，表示函数参数的类型列表，用于传递给funPtr。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  参数说明

<a name="table1055216132132"></a>
<table><thead align="left"><tr id="row105531513121315"><th class="cellrowborder" valign="top" width="16.49%" id="mcps1.2.4.1.1"><p id="p5553171319138"><a name="p5553171319138"></a><a name="p5553171319138"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="11.93%" id="mcps1.2.4.1.2"><p id="p5553151313131"><a name="p5553151313131"></a><a name="p5553151313131"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="71.58%" id="mcps1.2.4.1.3"><p id="p655316136139"><a name="p655316136139"></a><a name="p655316136139"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row466617486269"><td class="cellrowborder" valign="top" width="16.49%" headers="mcps1.2.4.1.1 "><p id="p566616486266"><a name="p566616486266"></a><a name="p566616486266"></a>Args... args</p>
</td>
<td class="cellrowborder" valign="top" width="11.93%" headers="mcps1.2.4.1.2 "><p id="p6666184812269"><a name="p6666184812269"></a><a name="p6666184812269"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.58%" headers="mcps1.2.4.1.3 "><p id="p1766614811264"><a name="p1766614811264"></a><a name="p1766614811264"></a>与class... Args对应的参数列表，表示传递给funPtr的实际参数。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section14483414194"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

无

## 调用示例<a name="section176061616102911"></a>

```
extern "C" __global__ __aicore__ void baremix_custom(GM_ADDR a, GM_ADDR b, GM_ADDR bias, GM_ADDR c,
                                                              GM_ADDR workspace, GM_ADDR tilingGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::TPipe pipe;
    TCubeTiling tiling;
    CopyTiling(&tiling, tilingGm);
    // 避免代码中直接的硬件条件判断（如使用ASCEND_IS_AIV或ASCEND_IS_AIC）
    Async<EngineType::AIC, aicOperation>(a, b, bias, c, workspace, tiling, &pipe);
    Async<EngineType::AIV, aivOperation>(c, tiling, &pipe);
}
__aicore__ inline void aicOperation(GM_ADDR a, GM_ADDR b, GM_ADDR bias, GM_ADDR c, GM_ADDR workspace, const TCubeTiling &tiling, AscendC::TPipe *pipe) {
    MatmulLeakyKernel<half, half, float, float> matmulLeakyKernel;
    matmulLeakyKernel.Init(a, b, bias, c, workspace, tiling, pipe);
    REGIST_MATMUL_OBJ(pipe, GetSysWorkSpacePtr(), matmulLeakyKernel.matmulObj, &matmulLeakyKernel.tiling);
    matmulLeakyKernel.Process(pipe);
}

__aicore__ inline void aivOperation(GM_ADDR c, const TCubeTiling &tiling, AscendC::TPipe *pipe) {
    LeakyReluKernel<float> leakyReluKernel;
    leakyReluKernel.Init(c, tiling, pipe);
    leakyReluKernel.Process(pipe);
}
```

