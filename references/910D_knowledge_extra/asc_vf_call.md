# asc\_vf\_call<a name="ZH-CN_TOPIC_0000002523304632"></a>

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

在SIMD与SIMT混合编程场景，启动SIMT VF（Vector Function）子任务，通过参数配置，启动指定数目的线程，执行指定的SIMT核函数。

> **说明：** 
>asc\_vf\_call启动SIMT VF子任务时，子任务函数不能是类的成员函数，推荐使用普通函数或类静态函数，且入口函数必须使用\_\_simt\_vf\_\_修饰宏。
>asc\_vf\_call启动SIMT VF子任务时，传递的参数只支持裸指针，常见基本数据类型。不支持传递结构体，数组等。

## 函数原型<a name="section620mcpsimp"></a>

```
template <auto funcPtr, typename... Args>
__aicore__ inline void asc_vf_call(dim3 threadNums, Args &&...args)
```

## 参数说明<a name="section0866173114710"></a>

**表 1**  模板参数说明

<a name="table4835205712588"></a>
<table><thead align="left"><tr id="row118356578583"><th class="cellrowborder" valign="top" width="23.549999999999997%" id="mcps1.2.3.1.1"><p id="p48354572582"><a name="p48354572582"></a><a name="p48354572582"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="76.44999999999999%" id="mcps1.2.3.1.2"><p id="p583535795817"><a name="p583535795817"></a><a name="p583535795817"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row1835857145817"><td class="cellrowborder" valign="top" width="23.549999999999997%" headers="mcps1.2.3.1.1 "><p id="p5835457165816"><a name="p5835457165816"></a><a name="p5835457165816"></a>funcPtr</p>
</td>
<td class="cellrowborder" valign="top" width="76.44999999999999%" headers="mcps1.2.3.1.2 "><p id="p206477515192"><a name="p206477515192"></a><a name="p206477515192"></a>用于指定SIMT入口核函数。</p>
</td>
</tr>
<tr id="row15761730201211"><td class="cellrowborder" valign="top" width="23.549999999999997%" headers="mcps1.2.3.1.1 "><p id="p16761030171214"><a name="p16761030171214"></a><a name="p16761030171214"></a>Args</p>
</td>
<td class="cellrowborder" valign="top" width="76.44999999999999%" headers="mcps1.2.3.1.2 "><p id="p1276123018129"><a name="p1276123018129"></a><a name="p1276123018129"></a>定义可变参数，用于传递实参到SIMT入口核函数。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  参数说明

<a name="zh-cn_topic_0235751031_table33761356"></a>
<table><thead align="left"><tr id="zh-cn_topic_0235751031_row27598891"><th class="cellrowborder" valign="top" width="14.12%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="14.469999999999999%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="71.41%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row42461942101815"><td class="cellrowborder" valign="top" width="14.12%" headers="mcps1.2.4.1.1 "><p id="p284425844311"><a name="p284425844311"></a><a name="p284425844311"></a>threadNums</p>
</td>
<td class="cellrowborder" valign="top" width="14.469999999999999%" headers="mcps1.2.4.1.2 "><p id="p158449584436"><a name="p158449584436"></a><a name="p158449584436"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p1378522191610"><a name="p1378522191610"></a><a name="p1378522191610"></a>dim3结构，定义为{dimx，dimy，dimz}，用于指定SIMT线程块内线程数量。线程总数为dimx * dimy * dimz，该值的大小必须小于等于2048。</p>
</td>
</tr>
<tr id="row625424812813"><td class="cellrowborder" valign="top" width="14.12%" headers="mcps1.2.4.1.1 "><p id="p6255648487"><a name="p6255648487"></a><a name="p6255648487"></a>args</p>
</td>
<td class="cellrowborder" valign="top" width="14.469999999999999%" headers="mcps1.2.4.1.2 "><p id="p182559481483"><a name="p182559481483"></a><a name="p182559481483"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p1832914611154"><a name="p1832914611154"></a><a name="p1832914611154"></a>可变参数，用于传递实参到SIMT入口核函数。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无

## 需要包含的头文件<a name="section7131139151810"></a>

使用该接口需要包含"simt\_api/common\_functions.h"头文件。

```
#include "simt_api/common_functions.h"
```

## 调用示例<a name="section1316724610428"></a>

对Global Memory数据做加法计算。

```
__simt_vf__ __launch_bounds__(2048) inline void SimtCompute(
    __gm__ float* dst, __gm__ float* src0, __gm__ float* src1, int count) const
{
    // simt 代码
    for(int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < count; idx += gridDim.x * blockDim.x)
    {
        dst[idx] = src0[idx] + src1[idx];
    }
}

__global__ __aicore__ void SimtComputeShell(__gm__ float* x, __gm__ float* y, __gm__ float* z, const int size)
{
    __gm__ float* dst = x;
    __gm__ float* src0 = y;
    __gm__ float* src1 = z;
    // asc_vf_call启动SIMT VF子任务
    asc_vf_call<SimtCompute>(dim3{1024, 1, 1}, dst, src0, src1, size);
}
```

