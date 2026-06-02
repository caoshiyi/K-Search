# assert<a name="ZH-CN_TOPIC_0000002523344786"></a>

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

## 功能说明<a name="section259105813316"></a>

本接口在SIMT VF调试场景下提供assert断言功能。在算子Kernel侧的SIMT VF实现代码中，如果assert的内部条件判断不为真，则会输出assert条件，并将输入的信息格式化打印在屏幕上，同时算子运行失败。

在算子Kernel侧代码的适当位置使用assert进行断言检查，并格式化输出一些调试信息。示例如下：

```
int assertFlag = 10;

assert(assertFlag == 10);
```

打印信息示例如下：

```
[ASSERT] /home/.../add_custom.cpp:44: : Assertion `assertFlag != 10' failed.
```

请注意，assert接口的打印功能对算子的实际运行性能有影响。

## 函数原型<a name="section2067518173415"></a>

```
assert(expr)
```

## 参数说明<a name="section158061867342"></a>

<a name="zh-cn_topic_0235751031_table33761356"></a>
<table><thead align="left"><tr id="zh-cn_topic_0235751031_row27598891"><th class="cellrowborder" valign="top" width="16.49%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="11.93%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="71.58%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row42461942101815"><td class="cellrowborder" valign="top" width="16.49%" headers="mcps1.1.4.1.1 "><p id="p541413413465"><a name="p541413413465"></a><a name="p541413413465"></a>expr</p>
</td>
<td class="cellrowborder" valign="top" width="11.93%" headers="mcps1.1.4.1.2 "><p id="p1441334144620"><a name="p1441334144620"></a><a name="p1441334144620"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.58%" headers="mcps1.1.4.1.3 "><p id="p84131146466"><a name="p84131146466"></a><a name="p84131146466"></a>assert断言是否终止程序的条件。条件为true则程序继续执行，条件为false则终止程序。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section43265506459"></a>

该接口当前仅支持融合编译场景。

## 需要包含的头文件<a name="section10354115115916"></a>

使用该接口需要包含"utils/debug/asc\_assert.h"头文件。

```
#include "utils/debug/asc_assert.h"
```

## 调用示例<a name="section82241477610"></a>

```
__simt_vf__ __launch_bounds__(1024) inline void AddCustom(__gm__ bool* dst, __gm__ float* x)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    assert(!isnan(x[idx]));
    dst[idx] = x[idx];
}
```

程序运行时会触发assert，打印效果如下：

```
[ASSERT] /home/.../add_custom.cpp:44: : Assertion `!isnan(x[idx])' failed.
```

