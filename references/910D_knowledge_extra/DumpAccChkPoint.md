# DumpAccChkPoint<a name="ZH-CN_TOPIC_0000002554424647"></a>

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
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p9269533102319"><a name="p9269533102319"></a><a name="p9269533102319"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section259105813316"></a>

基于算子工程开发的算子，可以使用该接口Dump指定Tensor的内容。同时支持打印自定义的附加信息（仅支持uint32\_t数据类型的信息），比如打印当前行号等。区别于[DumpTensor](DumpTensor.md)，使用该接口可以支持指定偏移位置的Tensor打印。

在算子kernel侧实现代码中需要打印偏移后Tensor数据的地方调用DumpAccChkPoint接口打印相关内容。样例如下：

```
AscendC::DumpAccChkPoint(srcLocal, 5, 32, dataLen);
```

> **注意：** 
>DumpAccChkPoint接口打印功能会对算子实际运行的性能带来一定影响，通常在调测阶段使用。开发者可以按需通过设置ASCENDC\_DUMP=0来关闭打印功能。

## 函数原型<a name="section2067518173415"></a>

```
template <typename T>
__aicore__ inline void DumpAccChkPoint(const LocalTensor<T> &tensor, uint32_t index, uint32_t countOff, uint32_t dumpSize)
template <typename T>
__aicore__ inline void DumpAccChkPoint(const GlobalTensor<T> &tensor, uint32_t index, uint32_t countOff, uint32_t dumpSize)
```

## 参数说明<a name="section158061867342"></a>

**表 1**  模板参数说明

<a name="table7228470519"></a>
<table><thead align="left"><tr id="row10228177350"><th class="cellrowborder" valign="top" width="16.42%" id="mcps1.2.3.1.1"><p id="p8228371856"><a name="p8228371856"></a><a name="p8228371856"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="83.58%" id="mcps1.2.3.1.2"><p id="p17228157653"><a name="p17228157653"></a><a name="p17228157653"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row822877953"><td class="cellrowborder" valign="top" width="16.42%" headers="mcps1.2.3.1.1 "><p id="p13228470510"><a name="p13228470510"></a><a name="p13228470510"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="83.58%" headers="mcps1.2.3.1.2 "><p id="p32281779511"><a name="p32281779511"></a><a name="p32281779511"></a>需要dump的Tensor的数据类型。</p>
<p id="p115681232195813"><a name="p115681232195813"></a><a name="p115681232195813"></a><span id="ph55691332205813"><a name="ph55691332205813"></a><a name="ph55691332205813"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：bool、uint8_t、int8_t、int16_t、uint16_t、int32_t、uint32_t、int64_t、uint64_t、float、half、bfloat16_t。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  参数说明

<a name="zh-cn_topic_0235751031_table33761356"></a>
<table><thead align="left"><tr id="zh-cn_topic_0235751031_row27598891"><th class="cellrowborder" valign="top" width="16.48%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="11.940000000000001%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="71.58%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row42461942101815"><td class="cellrowborder" valign="top" width="16.48%" headers="mcps1.2.4.1.1 "><p id="p1852414323610"><a name="p1852414323610"></a><a name="p1852414323610"></a>tensor</p>
</td>
<td class="cellrowborder" valign="top" width="11.940000000000001%" headers="mcps1.2.4.1.2 "><p id="p17524104318366"><a name="p17524104318366"></a><a name="p17524104318366"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.58%" headers="mcps1.2.4.1.3 "><p id="p750413437368"><a name="p750413437368"></a><a name="p750413437368"></a>需要dump的Tensor。</p>
<p id="p975712273273"><a name="p975712273273"></a><a name="p975712273273"></a>待dump的tensor位于<span id="ph33931718291"><a name="ph33931718291"></a><a name="ph33931718291"></a>Unified Buffer</span>/<span id="ph1439317732913"><a name="ph1439317732913"></a><a name="ph1439317732913"></a>L1 Buffer</span>/<span id="ph133937714292"><a name="ph133937714292"></a><a name="ph133937714292"></a>L0C Buffer</span>时使用LocalTensor类型的tensor参数输入。</p>
<p id="p614319132295"><a name="p614319132295"></a><a name="p614319132295"></a>待dump的tensor位于<span id="ph9775012163018"><a name="ph9775012163018"></a><a name="ph9775012163018"></a>Global Memory</span>时使用GlobalTensor类型的tensor参数输入。</p>
</td>
</tr>
<tr id="row241512381322"><td class="cellrowborder" valign="top" width="16.48%" headers="mcps1.2.4.1.1 "><p id="p549445182514"><a name="p549445182514"></a><a name="p549445182514"></a>index</p>
</td>
<td class="cellrowborder" valign="top" width="11.940000000000001%" headers="mcps1.2.4.1.2 "><p id="p16415538133215"><a name="p16415538133215"></a><a name="p16415538133215"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.58%" headers="mcps1.2.4.1.3 "><p id="p10415193810325"><a name="p10415193810325"></a><a name="p10415193810325"></a>用户自定义附加信息（行号或其他自定义数字）。</p>
</td>
</tr>
<tr id="row619402411145"><td class="cellrowborder" valign="top" width="16.48%" headers="mcps1.2.4.1.1 "><p id="p1161414475568"><a name="p1161414475568"></a><a name="p1161414475568"></a>countOff</p>
</td>
<td class="cellrowborder" valign="top" width="11.940000000000001%" headers="mcps1.2.4.1.2 "><p id="p1091852195615"><a name="p1091852195615"></a><a name="p1091852195615"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.58%" headers="mcps1.2.4.1.3 "><p id="p280391985610"><a name="p280391985610"></a><a name="p280391985610"></a>偏移元素个数。偏移后的Tensor地址需要满足所在物理位置的对齐约束，具体参考<a href="通用说明和约束.md">通用说明和约束</a>。</p>
</td>
</tr>
<tr id="row0376640163215"><td class="cellrowborder" valign="top" width="16.48%" headers="mcps1.2.4.1.1 "><p id="p437715407324"><a name="p437715407324"></a><a name="p437715407324"></a>dumpSize</p>
</td>
<td class="cellrowborder" valign="top" width="11.940000000000001%" headers="mcps1.2.4.1.2 "><p id="p1337712403326"><a name="p1337712403326"></a><a name="p1337712403326"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.58%" headers="mcps1.2.4.1.3 "><p id="p7377104073217"><a name="p7377104073217"></a><a name="p7377104073217"></a>需要dump的元素个数。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section794123819592"></a>

-   该功能仅用于NPU上板调试。
-   暂不支持算子入图场景的打印。

-   当前仅支持打印存储位置为Unified Buffer/L1 Buffer/L0C Buffer/Global Memory的Tensor信息。针对Ascend 950PR/Ascend 950DT，不支持打印L1 Buffer上的Tensor信息。
-   操作数地址对齐要求请参见[通用地址对齐约束](通用说明和约束.md#section796754519912)。
-   单次调用DumpTensor打印的数据总量不可超过1MB（还包括少量框架需要的头尾信息，通常可忽略）。使用时应注意，如果超出这个限制，则数据不会被打印。
-   在计算数据量时，若Dump的总长度未对齐，需要考虑padding数据的影响。当进行非对齐Dump时，如果实际Dump的元素长度不满足32字节对齐，系统会在其末尾自动补充一定数量的padding数据，以满足对齐要求。例如，Tensor1中用户需要Dump的元素长度为30字节，系统会在其后添加2字节的padding，使总长度对齐到32字节。但在实际解析时，仍只解析原始的30字节数据，padding部分不会被使用。
-   使用自定义算子工程进行算子开发时，接口的打印信息和上文描述有些差异：

    Dump时，每个block核的dump信息前会增加对应信息头DumpHead，用于记录核号和资源使用信息；每次Dump的Tensor数据前也会添加信息头DumpTensorHead，用于记录Tensor的相关信息。如下图所示，展示了多核打印场景下的打印信息结构。

    <!-- img2text -->
```
block0
┌──────────┬────────────────┬────────┬────────────────┬────────┬────────────────┬─────┐
│ DumpHead │ DumpTensorHead │ Tensor1│ DumpTensorHead │ Tensor2│ DumpTensorHead │ ... │
└──────────┴────────────────┴────────┴────────────────┴────────┴────────────────┴─────┘

block1
┌──────────┬────────────────┬────────┬────────────────┬────────┬────────────────┬─────┐
│ DumpHead │ DumpTensorHead │ Tensor1│ DumpTensorHead │ Tensor2│ DumpTensorHead │ ... │
└──────────┴────────────────┴────────┴────────────────┴────────┴────────────────┴─────┘

                                  ...

blockn
┌──────────┬────────────────┬────────┬────────────────┬────────┬────────────────┬─────┐
│ DumpHead │ DumpTensorHead │ Tensor1│ DumpTensorHead │ Tensor2│ DumpTensorHead │ ... │
└──────────┴────────────────┴────────┴────────────────┴────────┴────────────────┴─────┘
```

    **DumpHead的具体信息如下：**

    -   opType：当前运行的算子类型；
    -   CoreType：当前运行的核的类型；
    -   block dim：开发者设置的算子执行核数；
    -   total\_block\_num：参与dump的核数；
    -   block\_remain\_len：当前核剩余可用的dump的空间；
    -   block\_initial\_space：当前核初始分配的dump空间；
    -   rsv：保留字段；
    -   magic：内存校验魔术字。

    DumpHead打印时，除了上述打印还会自动打印当前所运行核的类型及对应的该类型下的核索引，如：AIV-0。

    **DumpTensorHead的具体信息如下：**

    -   desc：用户自定义附加信息；
    -   addr：Tensor的地址；
    -   data\_type：Tensor的数据类型；
    -   position：表示Tensor所在的物理存储位置，当前仅支持Unified Buffer/L1 Buffer/L0C Buffer/Global Memory。
    -   dump\_size：表示用户需要dump的元素个数。

    DumpAccChkPoint打印结果的最前面会自动打印CANN\_VERSION\_STR值与CANN\_TIMESTAMP值。其中，CANN\_VERSION\_STR与CANN\_TIMESTAMP为宏定义，CANN\_VERSION\_STR代表CANN软件包的版本号信息，形式为字符串，CANN\_TIMESTAMP为CANN软件包发布时的时间戳，形式为数值（uint64\_t）。开发者也可在代码中直接使用这两个宏。

    打印结果的样例如下：

    ```
    opType=AddCustom, DumpHead: AIV-0, CoreType=AIV, block dim=8, total_block_num=8, block_remain_len=1046912, block_initial_space=1048576, rsv=0, magic=5aa5bccd 
    CANN Version: XX.XX,TimeStamp: XXXXXX
    DumpTensor: desc=5, addr=40, data_type=float16, position=UB, dump_size=32
    [16.000000, 22.000000, 2.000000, 3.000000, 58.000000, 62.000000, 33.000000, 74.000000, 51.000000, 69.000000, 61.000000, 9.000000, 53.000000, 35.000000, 14.000000, 43.000000, 20.000000, 43.000000, 92.000000, 84.000000, 9.000000, 6.000000, 78.000000, 53.000000, 52.000000, 33.000000, 51.000000, 61.000000, 92.000000, 45.000000, 39.000000,34.000000]
    ...
    DumpTensor: desc=5, addr=140, data_type=float16, position=UB, dump_size=32
    [41.000000, 91.000000, 12.000000, 32.000000, 28.000000, 49.000000, 2.000000, 75.000000, 11.000000, 32.000000, 17.000000, 31.000000, 70.000000, 38.000000, 76.000000, 87.000000, 61.000000, 8.000000, 55.000000, 70.000000, 17.000000, 37.000000, 35.000000, 58.000000, 94.000000, 31.000000, 50.000000, 29.000000, 13.000000, 37.000000, 79.000000,29.000000]
    ```

    该接口使用Dump功能，一个算子所有使用Dump功能的接口在每个核上Dump的数据总量不可超过1M。请开发者自行控制待打印的内容数据量，超出则不会打印。

## 调用示例<a name="section82241477610"></a>

```
constexpr uint32_t totalLength = 256;    // 参与搬运的元素个数
AscendC::LocalTensor<half> srcLocal;
AscendC::GlobalTensor<half> srcGlobal;
AscendC::DataCopy(srcLocal, srcGlobal, totalLength * sizeof(half));
uint32_t index = 8;    // 用户自定义附加信息，此处传入DumpAccChkPoint指令的行号
uint32_t countOff = 32;    // 偏移元素个数，从srcLocal[32]开始打印
uint32_t dupmSize = 128;    // dump的元素个数，从srcLocal[32]开始打印128个元素个数
AscendC::DumpAccChkPoint(srcLocal, index, countOff, dupmSize);
```

