# \_\_float22half2\_rz\_sat<a name="ZH-CN_TOPIC_0000002523303846"></a>

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

饱和模式下，将float2类型数据的两个分量遵循CAST\_TRUNC模式转换为半精度浮点数，返回转换后的half2类型数据。

## 函数原型<a name="section620mcpsimp"></a>

```
__simt_callee__ inline half2 __float22half2_rz_sat(const float2 x)
```

## 参数说明<a name="section0866173114710"></a>

**表 1**  参数说明

<a name="zh-cn_topic_0235751031_table33761356"></a>
<table><thead align="left"><tr id="zh-cn_topic_0235751031_row27598891"><th class="cellrowborder" valign="top" width="18.54%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="10.05%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="71.41%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row42461942101815"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p284425844311"><a name="p284425844311"></a><a name="p284425844311"></a>x</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p158449584436"><a name="p158449584436"></a><a name="p158449584436"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p6389114616514"><a name="p6389114616514"></a><a name="p6389114616514"></a>源操作数。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

饱和模式下将输入的两个分量遵循CAST\_TRUNC模式转换成的half2类型数据。

## 约束说明<a name="section633mcpsimp"></a>

使用此接口前需将CTRL\[60\]寄存器设置为0，否则饱和模式不生效。设置方式请参见[控制饱和行为的方式](简介-162.md#section1194916101549)。

## 需要包含的头文件<a name="section7131139151810"></a>

使用该接口需要包含"simt\_api/asc\_fp16.h"头文件。

```
#include "simt_api/asc_fp16.h"
```

## 调用示例<a name="section134121647154719"></a>

```
// 使用小向量可提升数据搬运效率
__simt_vf__ __launch_bounds__(1024) inline void simt_float22half2_rz_sat(__gm__ float2* input, __gm__ half2* output, uint32_t input_total_length)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    // 每个线程处理1个float2类型的数据，即2个float类型的数据，因此idx >= input_total_length / 2的线程不处理数据
    if (idx >= input_total_length / 2) {
        return;
    }
    output[idx] = __float22half2_rz_sat(input[idx]);
}

__global__ __vector__ void cast_kernel(__gm__ float* input, __gm__ half* output, uint32_t input_total_length)
{
    asc_vf_call<simt_float22half2_rz_sat>(dim3(1024), (__gm__ float2*)input, (__gm__ half2*)output, input_total_length);
}
```

