# GetBlockIdx<a name="ZH-CN_TOPIC_0000002554423627"></a>

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

获取当前核的index，用于代码内部的多核逻辑控制及多核偏移量计算等。

## 函数原型<a name="section620mcpsimp"></a>

```
__aicore__ inline int64_t GetBlockIdx()
```

## 参数说明<a name="section622mcpsimp"></a>

无

## 返回值说明<a name="section640mcpsimp"></a>

当前核的index。

index的范围为\[0, 用户配置的NumBlocks数量 - 1\]。

## 约束说明<a name="section633mcpsimp"></a>

GetBlockIdx为一个系统内置函数，返回当前核的index。

## 调用示例<a name="section177231425115410"></a>

```
// srcGm、dstGm为外部输入的gm空间
AscendC::GlobalTensor<float> srcGlobal;
AscendC::GlobalTensor<float> dstGlobal;
blockNum = AscendC::GetBlockNum(); // 获取核总数
perBlockSize = srcDataSize / blockNum; // 每个核平分处理相同个数
blockIdx = AscendC::GetBlockIdx(); // 获取当前工作的核ID
srcGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(srcGm + blockIdx * perBlockSize * sizeof(float)), perBlockSize);    // 分配每个核上的srcGlobal的内存地址
dstGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(dstGm + blockIdx * perBlockSize * sizeof(float)), perBlockSize);    // 分配每个核上的dstGlobal的内存地址
```

