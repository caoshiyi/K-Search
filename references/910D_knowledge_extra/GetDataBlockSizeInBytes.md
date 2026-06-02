# GetDataBlockSizeInBytes<a name="ZH-CN_TOPIC_0000002554424787"></a>

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

获取当前芯片版本一个datablock的大小，单位为byte。开发者根据datablock的大小来计算API指令中待传入的repeatTime 、DataBlock Stride、Repeat Stride等参数值。

## 函数原型<a name="section620mcpsimp"></a>

```
__aicore__ inline constexpr int16_t GetDataBlockSizeInBytes()
```

## 参数说明<a name="section622mcpsimp"></a>

无

## 返回值说明<a name="section640mcpsimp"></a>

当前芯片版本一个datablock的大小，单位为byte。

## 约束说明<a name="section633mcpsimp"></a>

无

## 调用示例<a name="section177231425115410"></a>

如下样例通过GetDataBlockSizeInBytes获取的datablock值，来计算repeatTime的值：

```
int16_t dataBlockSize = AscendC::GetDataBlockSizeInBytes();
// 每个repeat有8个datablock,可计算8 * dataBlockSize / sizeof(half)个数，mask配置为迭代内所有元素均参与计算
uint64_t mask = 8 * dataBlockSize / sizeof(half);
// 共计算512个数，除以每个repeat参与计算的元素个数，得到repeatTime
uint8_t repeatTime = 512 / mask; 
// dstBlkStride, src0BlkStride, src1BlkStride = 1, 单次迭代内数据连续读取和写入
// dstRepStride, src0RepStride, src1RepStride = 8, 相邻迭代间数据连续读取和写入
AscendC::Add(dstLocal, src0Local, src1Local, mask, repeatTime, { 1, 1, 1, 8, 8, 8 });
```

