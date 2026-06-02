# GetUBSizeInBytes<a name="ZH-CN_TOPIC_0000002523304980"></a>

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

获取UB空间的大小，单位为byte。开发者根据UB的大小来计算循环次数等参数值。

## 函数原型<a name="section620mcpsimp"></a>

```
__aicore__ inline constexpr uint32_t GetUBSizeInBytes()
```

## 参数说明<a name="section622mcpsimp"></a>

无

## 返回值说明<a name="section640mcpsimp"></a>

UB空间的大小，单位为byte。

## 约束说明<a name="section633mcpsimp"></a>

无

## 调用示例<a name="section177231425115410"></a>

本调用示例通过GetUBSizeInBytes获取的UB空间大小，来计算tileNum的值。完整的算子样例请参考：[算子样例](https://gitcode.com/cann/asc-devkit/tree/master/examples/02_features/15_utility_function/get_ub_size_in_bytes)。

```
#include "kernel_operator.h"

uint32_t totalLength = 16384;
uint32_t tileLength = AscendC::GetUBSizeInBytes() / sizeof(half) / 2;
if (totalLength < tileLength) {
    tileLength = totalLength;
}
uint32_t tileNum = totalLength / tileLength;
```

