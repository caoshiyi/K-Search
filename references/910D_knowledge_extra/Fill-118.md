# Fill<a name="ZH-CN_TOPIC_0000002554424535"></a>

## 产品支持情况<a name="section1586581915393"></a>

<a name="table169596713360"></a>
<table><thead align="left"><tr id="row129590715369"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p17959971362"><a name="p17959971362"></a><a name="p17959971362"></a><span id="ph895914718367"><a name="ph895914718367"></a><a name="ph895914718367"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p89594763612"><a name="p89594763612"></a><a name="p89594763612"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row18959673369"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p1595910763613"><a name="p1595910763613"></a><a name="p1595910763613"></a><span id="ph1595918753613"><a name="ph1595918753613"></a><a name="ph1595918753613"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p1695957133611"><a name="p1695957133611"></a><a name="p1695957133611"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section618mcpsimp"></a>

将Global Memory上的数据初始化为指定值。该接口可用于对workspace地址或输出数据进行清零。

## 函数原型<a name="section620mcpsimp"></a>

```
template <typename T>
__aicore__ inline void Fill(GlobalTensor<T>& gmWorkspaceAddr, const uint64_t size, const T value)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table575571914269"></a>
<table><thead align="left"><tr id="row18755131942614"><th class="cellrowborder" valign="top" width="19.39%" id="mcps1.2.3.1.1"><p id="p675519193268"><a name="p675519193268"></a><a name="p675519193268"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="80.61%" id="mcps1.2.3.1.2"><p id="p2258534164211"><a name="p2258534164211"></a><a name="p2258534164211"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row14755141911264"><td class="cellrowborder" valign="top" width="19.39%" headers="mcps1.2.3.1.1 "><p id="p47551198266"><a name="p47551198266"></a><a name="p47551198266"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="80.61%" headers="mcps1.2.3.1.2 "><p id="p125969172719"><a name="p125969172719"></a><a name="p125969172719"></a>操作数的数据类型。</p>
<p id="p153204151019"><a name="p153204151019"></a><a name="p153204151019"></a><span id="ph133209151609"><a name="ph133209151609"></a><a name="ph133209151609"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：uint8_t、int8_t、uint16_t、int16_t、bfloat16_t、half、uint32_t、int32_t、float、uint64_t、int64_t。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  接口参数说明

<a name="table62161631132810"></a>
<table><thead align="left"><tr id="row12216103118284"><th class="cellrowborder" valign="top" width="13.661366136613662%" id="mcps1.2.4.1.1"><p id="p1421643114288"><a name="p1421643114288"></a><a name="p1421643114288"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="12.591259125912593%" id="mcps1.2.4.1.2"><p id="p82165310285"><a name="p82165310285"></a><a name="p82165310285"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="73.74737473747375%" id="mcps1.2.4.1.3"><p id="p1121663111288"><a name="p1121663111288"></a><a name="p1121663111288"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row82161131182810"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p248574352318"><a name="p248574352318"></a><a name="p248574352318"></a>gmWorkspaceAddr</p>
</td>
<td class="cellrowborder" valign="top" width="12.591259125912593%" headers="mcps1.2.4.1.2 "><p id="p144841743122315"><a name="p144841743122315"></a><a name="p144841743122315"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.74737473747375%" headers="mcps1.2.4.1.3 "><p id="p1483143162317"><a name="p1483143162317"></a><a name="p1483143162317"></a>gmWorkspaceAddr为用户定义的全局Global空间，是需要被初始化的空间，类型为GlobalTensor。GlobalTensor数据结构的定义请参考<a href="GlobalTensor.md">GlobalTensor</a>。</p>
</td>
</tr>
<tr id="row5216163192815"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p164821543202310"><a name="p164821543202310"></a><a name="p164821543202310"></a>size</p>
</td>
<td class="cellrowborder" valign="top" width="12.591259125912593%" headers="mcps1.2.4.1.2 "><p id="p64811843112313"><a name="p64811843112313"></a><a name="p64811843112313"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.74737473747375%" headers="mcps1.2.4.1.3 "><p id="p4481643182317"><a name="p4481643182317"></a><a name="p4481643182317"></a>需要初始化的空间大小，单位为元素个数。</p>
</td>
</tr>
<tr id="row1212625414239"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p312613545238"><a name="p312613545238"></a><a name="p312613545238"></a>value</p>
</td>
<td class="cellrowborder" valign="top" width="12.591259125912593%" headers="mcps1.2.4.1.2 "><p id="p10126254182313"><a name="p10126254182313"></a><a name="p10126254182313"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.74737473747375%" headers="mcps1.2.4.1.3 "><p id="p51261854122320"><a name="p51261854122320"></a><a name="p51261854122320"></a>初始化的值，数据类型与gmWorkspaceAddr保持一致。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section91032023123812"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

-   单核调用此接口时，如果后续操作涉及Unified Buffer的使用，则需要在调用接口后，设置MTE2流水等待MTE3流水（[MTE3\_MTE2](SetFlag-WaitFlag(ISASI).md#section622mcpsimp)）的同步。
-   当多个核调用此接口对Global Memory进行初始化时，所有核对Global Memory的初始化未必会同时结束，也可能存在核之间读后写、写后读以及写后写等数据依赖问题。这种使用场景下，可以在本接口后调用[SyncAll](SyncAll.md)接口保证多核间同步正确。
-   该接口仅支持在程序内存分配[InitBuffer](InitBuffer.md)接口前使用。

## 调用示例<a name="section642mcpsimp"></a>

本调用示例使用8个核，每个核用当前blockIdx的值初始化zGm上的65536个数，每个核的核内计算为x和y两组全1的65536个half类型数据相加，计算结果累加到zGm。此样例中8个核的blockIdx分别为0到7，输入x和y均为全1数据，则最终zGm输出数据为2到9。完整的算子样例请参考[fill算子样例](https://gitcode.com/cann/asc-devkit/tree/master/examples/03_libraries/09_transpose/fill)。

```
#include "kernel_operator.h"

constexpr int32_t INIT_SIZE = 65536;

zGm.SetGlobalBuffer((__gm__ float*)z + INIT_SIZE * AscendC::GetBlockIdx(), INIT_SIZE);
AscendC::Fill(zGm, INIT_SIZE, (float)(AscendC::GetBlockIdx()));
```

结果示例如下：

```
输入数据(x):
[1. 1. 1. 1. 1. ... 1.]
输入数据(y):
[1. 1. 1. 1. 1. ... 1.]
输出数据(z):
[2. 2. 2. 2. 2. ... 2.
3. 3. 3. 3. 3. ... 3.
4. 4. 4. 4. 4. ... 4.
5. 5. 5. 5. 5. ... 5.
6. 6. 6. 6. 6. ... 6.
7. 7. 7. 7. 7. ... 7.
8. 8. 8. 8. 8. ... 8.
9. 9. 9. 9. 9. ... 9.]
```

