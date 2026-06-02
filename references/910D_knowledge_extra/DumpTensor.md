# DumpTensor<a name="ZH-CN_TOPIC_0000002554423467"></a>

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

基于算子工程开发的算子，可以使用该接口Dump指定Tensor的内容。同时支持打印自定义的附加信息（仅支持uint32\_t数据类型的信息），比如打印当前行号等。

在算子kernel侧实现代码中需要打印Tensor数据的地方调用DumpTensor接口打印相关内容。样例如下：

```
AscendC::DumpTensor(srcLocal, 5, dataLen);
```

> **注意：** 
>DumpTensor接口打印功能会对算子实际运行的性能带来一定影响，通常在调测阶段使用。开发者可以按需通过设置ASCENDC\_DUMP=0来关闭打印功能。

打印示例如下：

```
DumpTensor: desc=5, addr=0, data_type=float16, position=UB, dump_size=32
[19.000000, 4.000000, 38.000000, 50.000000, 39.000000, 67.000000, 84.000000, 98.000000, 21.000000, 36.000000, 18.000000, 46.000000, 10.000000, 92.000000, 26.000000, 38.000000, 39.000000, 9.000000, 82.000000, 37.000000, 35.000000, 65.000000, 97.000000, 59.000000, 89.000000, 63.000000, 70.000000, 57.000000, 35.000000, 3.000000, 16.000000,
42.000000]
DumpTensor: desc=5, addr=100, data_type=float16, position=UB, dump_size=32
[6.000000, 34.000000, 52.000000, 38.000000, 73.000000, 38.000000, 35.000000, 14.000000, 67.000000, 62.000000, 30.000000, 49.000000, 86.000000, 37.000000, 84.000000, 18.000000, 38.000000, 18.000000, 44.000000, 21.000000, 86.000000, 99.000000, 13.000000, 79.000000, 84.000000, 9.000000, 48.000000, 74.000000, 52.000000, 99.000000, 80.000000,
53.000000]
...
DumpTensor: desc=5, addr=0, data_type=float16, position=UB, dump_size=32
[35.000000, 41.000000, 41.000000, 22.000000, 84.000000, 49.000000, 60.000000, 0.000000, 90.000000, 14.000000, 67.000000, 80.000000, 16.000000, 46.000000, 16.000000, 83.000000, 6.000000, 70.000000, 97.000000, 28.000000, 97.000000, 62.000000, 80.000000, 22.000000, 53.000000, 37.000000, 23.000000, 58.000000, 65.000000, 28.000000, 4.000000,
29.000000]
```

## 函数原型<a name="section2067518173415"></a>

-   无Tensor shape的打印

    ```
    template <typename T>
    __aicore__ inline void DumpTensor(const LocalTensor<T> &tensor, uint32_t desc, uint32_t dumpSize)
    template <typename T>
    __aicore__ inline void DumpTensor(const GlobalTensor<T>& tensor, uint32_t desc, uint32_t dumpSize)
    ```

-   带Tensor shape的打印

    ```
    template <typename T>
    __aicore__ inline void DumpTensor(const LocalTensor<T>& tensor, uint32_t desc, uint32_t dumpSize, const ShapeInfo& shapeInfo)
    template <typename T>
    __aicore__ inline void DumpTensor(const GlobalTensor<T>& tensor, uint32_t desc, uint32_t dumpSize, const ShapeInfo& shapeInfo)
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
<p id="p33891341206"><a name="p33891341206"></a><a name="p33891341206"></a><span id="ph126252025205"><a name="ph126252025205"></a><a name="ph126252025205"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：bool、uint8_t、int8_t、int16_t、uint16_t、int32_t、uint32_t、int64_t、uint64_t、float、half、bfloat16_t、fp8_e4m3fn_t、fp8_e5m2_t、hifloat8_t、fp8_e8m0_t。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  参数说明

<a name="zh-cn_topic_0235751031_table33761356"></a>
<table><thead align="left"><tr id="zh-cn_topic_0235751031_row27598891"><th class="cellrowborder" valign="top" width="16.49%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="11.93%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="71.58%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row42461942101815"><td class="cellrowborder" valign="top" width="16.49%" headers="mcps1.2.4.1.1 "><p id="p1852414323610"><a name="p1852414323610"></a><a name="p1852414323610"></a>tensor</p>
</td>
<td class="cellrowborder" valign="top" width="11.93%" headers="mcps1.2.4.1.2 "><p id="p17524104318366"><a name="p17524104318366"></a><a name="p17524104318366"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.58%" headers="mcps1.2.4.1.3 "><p id="p750413437368"><a name="p750413437368"></a><a name="p750413437368"></a>需要dump的Tensor。</p>
<a name="ul332414592349"></a><a name="ul332414592349"></a><ul id="ul332414592349"><li>待dump的tensor位于<span id="ph33931718291"><a name="ph33931718291"></a><a name="ph33931718291"></a>Unified Buffer</span>/<span id="ph1439317732913"><a name="ph1439317732913"></a><a name="ph1439317732913"></a>L1 Buffer</span>/<span id="ph133937714292"><a name="ph133937714292"></a><a name="ph133937714292"></a>L0C Buffer</span>时使用LocalTensor类型的tensor参数输入。</li><li>待dump的tensor位于<span id="ph9775012163018"><a name="ph9775012163018"></a><a name="ph9775012163018"></a>Global Memory</span>时使用GlobalTensor类型的tensor参数输入。</li></ul>
</td>
</tr>
<tr id="row241512381322"><td class="cellrowborder" valign="top" width="16.49%" headers="mcps1.2.4.1.1 "><p id="p4415838153219"><a name="p4415838153219"></a><a name="p4415838153219"></a>desc</p>
</td>
<td class="cellrowborder" valign="top" width="11.93%" headers="mcps1.2.4.1.2 "><p id="p16415538133215"><a name="p16415538133215"></a><a name="p16415538133215"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.58%" headers="mcps1.2.4.1.3 "><p id="p10415193810325"><a name="p10415193810325"></a><a name="p10415193810325"></a>用户自定义附加信息（行号或其他自定义数字）。</p>
<p id="p1528621121415"><a name="p1528621121415"></a><a name="p1528621121415"></a><span>在使用DumpTensor功能时，用户可通过</span>desc<span>参数附加自定义信息，以便在不同调用场景下区分Dump内容的来源。此功能有助于精准定位具体DumpTensor的输出，提升调试与分析效率。</span></p>
</td>
</tr>
<tr id="row0376640163215"><td class="cellrowborder" valign="top" width="16.49%" headers="mcps1.2.4.1.1 "><p id="p437715407324"><a name="p437715407324"></a><a name="p437715407324"></a>dumpSize</p>
</td>
<td class="cellrowborder" valign="top" width="11.93%" headers="mcps1.2.4.1.2 "><p id="p1337712403326"><a name="p1337712403326"></a><a name="p1337712403326"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.58%" headers="mcps1.2.4.1.3 "><p id="p7377104073217"><a name="p7377104073217"></a><a name="p7377104073217"></a>需要dump的元素个数。</p>
</td>
</tr>
<tr id="row132761141132916"><td class="cellrowborder" valign="top" width="16.49%" headers="mcps1.2.4.1.1 "><p id="p5672172613207"><a name="p5672172613207"></a><a name="p5672172613207"></a>shapeInfo</p>
</td>
<td class="cellrowborder" valign="top" width="11.93%" headers="mcps1.2.4.1.2 "><p id="p1267217263206"><a name="p1267217263206"></a><a name="p1267217263206"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.58%" headers="mcps1.2.4.1.3 "><div class="p" id="p1367252616208"><a name="p1367252616208"></a><a name="p1367252616208"></a>传入Tensor的shape信息，可按照shape信息进行打印。<a name="ul1940123462918"></a><a name="ul1940123462918"></a><ul id="ul1940123462918"><li>当Shape尺寸大于dumpSize元素个数时， 按照ShapeInfo打印元素，不足的Dump数据用"-"展示。</li></ul>
<a name="ul619815578186"></a><a name="ul619815578186"></a><ul id="ul619815578186"><li>当Shape尺寸小于等于dumpSize元素个数时， 按照ShapeInfo打印元素，多出的Dump数据不展示。</li></ul>
</div>
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
┌──────────┬────────────────┬──────────┬────────────────┬──────────┬────────────────┬─────┐
│ DumpHead │ DumpTensorHead │ Tensor1  │ DumpTensorHead │ Tensor2  │ DumpTensorHead │ ... │
└──────────┴────────────────┴──────────┴────────────────┴──────────┴────────────────┴─────┘

block1
┌──────────┬────────────────┬──────────┬────────────────┬──────────┬────────────────┬─────┐
│ DumpHead │ DumpTensorHead │ Tensor1  │ DumpTensorHead │ Tensor2  │ DumpTensorHead │ ... │
└──────────┴────────────────┴──────────┴────────────────┴──────────┴────────────────┴─────┘

                                      ...

blockn
┌──────────┬────────────────┬──────────┬────────────────┬──────────┬────────────────┬─────┐
│ DumpHead │ DumpTensorHead │ Tensor1  │ DumpTensorHead │ Tensor2  │ DumpTensorHead │ ... │
└──────────┴────────────────┴──────────┴────────────────┴──────────┴────────────────┴─────┘
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
    -   position：表示Tensor所在的物理存储位置，当前仅支持Unified Buffer/L1 Buffer/L0C Buffer/Global Memory；
    -   dump\_size：表示用户需要dump的元素个数。

    DumpTensor打印结果的最前面会自动打印CANN\_VERSION\_STR值与CANN\_TIMESTAMP值。其中，CANN\_VERSION\_STR与CANN\_TIMESTAMP为宏定义，CANN\_VERSION\_STR代表CANN软件包的版本号信息，形式为字符串，CANN\_TIMESTAMP为CANN软件包发布时的时间戳，形式为数值（uint64\_t）。开发者也可在代码中直接使用这两个宏。

    打印示例如下：

    ```
    opType=AddCustom, DumpHead: AIV-0, CoreType=AIV, block dim=8, total_block_num=8, block_remain_len=1046912, block_initial_space=1048576, rsv=0, magic=5aa5bccd
    CANN Version: XX.XX, TimeStamp: XXXXXX
    DumpTensor: desc=5, addr=0, data_type=float16, position=UB, dump_size=32
    [19.000000, 4.000000, 38.000000, 50.000000, 39.000000, 67.000000, 84.000000, 98.000000, 21.000000, 36.000000, 18.000000, 46.000000, 10.000000, 92.000000, 26.000000, 38.000000, 39.000000, 9.000000, 82.000000, 37.000000, 35.000000, 65.000000, 97.000000, 59.000000, 89.000000, 63.000000, 70.000000, 57.000000, 35.000000, 3.000000, 16.000000,
    42.000000]
    DumpTensor: desc=5, addr=100, data_type=float16, position=UB, dump_size=32
    [6.000000, 34.000000, 52.000000, 38.000000, 73.000000, 38.000000, 35.000000, 14.000000, 67.000000, 62.000000, 30.000000, 49.000000, 86.000000, 37.000000, 84.000000, 18.000000, 38.000000, 18.000000, 44.000000, 21.000000, 86.000000, 99.000000, 13.000000, 79.000000, 84.000000, 9.000000, 48.000000, 74.000000, 52.000000, 99.000000, 80.000000,
    53.000000]
    ...
    DumpTensor: desc=5, addr=0, data_type=float16, position=UB, dump_size=32
    [35.000000, 41.000000, 41.000000, 22.000000, 84.000000, 49.000000, 60.000000, 0.000000, 90.000000, 14.000000, 67.000000, 80.000000, 16.000000, 46.000000, 16.000000, 83.000000, 6.000000, 70.000000, 97.000000, 28.000000, 97.000000, 62.000000, 80.000000, 22.000000, 53.000000, 37.000000, 23.000000, 58.000000, 65.000000, 28.000000, 4.000000,
    29.000000]
    ```

    该接口使用Dump功能，一个算子所有使用Dump功能的接口在每个核上Dump的数据总量（包括信息头）不可超过1M。请开发者自行控制待打印的内容数据量，超出则不会打印。

## 调用示例<a name="section82241477610"></a>

-   无Tensor shape的打印

    ```
    AscendC::DumpTensor(srcLocal, 5, dataLen);
    ```

-   带Tensor shape的打印

    ```
    uint32_t array[] = {static_cast<uint32_t>(8), static_cast<uint32_t>(8)};
    AscendC::ShapeInfo shapeInfo(2, array);       // dim为2， shape为(8,8)
    AscendC::DumpTensor(x, 2, 64, shapeInfo);     // dump x的64个元素，且解析按照shapeInfo的(8,8)排列
    
    uint32_t array1[] = {static_cast<uint32_t>(7), static_cast<uint32_t>(8)};
    AscendC::ShapeInfo shapeInfo1(2, array1); // dim为2， shape为(7,8)
    AscendC::DumpTensor(x1, 3, 64, shapeInfo1); // 当Shape尺寸小于等于dumpSize元素个数时， 按照ShapeInfo打印元素，多出的Dump数据不展示
    
    uint32_t array2[] = {static_cast<uint32_t>(9), static_cast<uint32_t>(8)};
    AscendC::ShapeInfo shapeInfo2(2, array2); // dim为2， shape为(9,8)
    AscendC::DumpTensor(x2, 4, 64, shapeInfo2); // 当Shape尺寸大于dumpSize元素个数时， 按照ShapeInfo打印元素，不足的Dump数据用"-"展示
    ```

    打印结果如下：

    ```
    DumpTensor: desc=2, addr=xxxx, data_type=float16, position=UB, dump_size=64
    [[150.000000,83.000000,109.000000,166.000000,129.000000,50.000000,150.000000,74.000000],
    [135.000000,79.000000,98.000000,134.000000,146.000000,166.000000,112.000000,70.000000],
    [122.000000,51.000000,116.000000,68.000000,172.000000,72.000000,102.000000,69.000000],
    [136.000000,83.000000,88.000000,88.000000,112.000000,148.000000,79.000000,136.000000],
    [133.000000,104.000000,83.000000,71.000000,83.000000,99.000000,103.000000,151.000000],
    [98.000000,118.000000,128.000000,83.000000,25.000000,105.000000,179.000000,34.000000],
    [104.000000,169.000000,115.000000,113.000000,134.000000,121.000000,88.000000,96.000000],
    [29.000000,139.000000,70.000000,40.000000,158.000000,138.000000,72.000000,171.000000]]
    DumpTensor: desc=3, addr=xxxx, data_type=float16, position=UB, dump_size=64
    shape is [7, 8], dumpSize is 64, dumpSize is greater than shapeSize.
    [[82.250000,37.312500,22.843750,91.937500,93.312500,77.125000,50.718750,27.171875],
    [21.859375,32.906250,20.109375,70.875000,13.398438,14.562500,30.156250,52.562500],
    [40.156250,45.781250,78.937500,65.687500,71.562500,61.375000,32.062500,80.750000],
    [55.593750,44.031250,43.781250,3.132812,38.750000,50.968750,79.562500,80.562500],
    [51.562500,22.468750,88.250000,20.578125,95.437500,83.562500,76.812500,34.281250],
    [75.500000,47.875000,52.562500,74.937500,39.687500,90.062500,28.890625,10.593750],
    [42.343750,67.062500,35.468750,60.875000,71.812500,81.562500,57.531250,62.500000]]
    DumpTensor: desc=4, addr=xxxx, data_type=float16, position=UB, dump_size=64
    shape is [9, 8], dumpSize is 64, data is not enough.
    [[95.437500,59.250000,57.281250,27.093750,41.375000,48.375000,33.093750,91.312500],
    [27.703125,60.718750,68.187500,70.875000,67.437500,84.562500,13.507812,4.550781],
    [24.500000,73.437500,36.062500,68.437500,55.500000,95.375000,60.250000,64.750000],
    [40.093750,85.000000,42.250000,39.531250,60.968750,8.953125,48.531250,53.906250],
    [53.656250,64.187500,84.750000,22.250000,95.500000,39.937500,12.945312,54.031250],
    [3.804688,98.187500,43.968750,26.000000,41.750000,34.500000,75.750000,89.625000],
    [25.046875,5.265625,65.500000,45.468750,32.937500,8.593750,1.705078,12.742188],
    [37.281250,95.125000,71.562500,27.515625,47.250000,36.312500,66.750000,31.250000],
    [-,-,-,-,-,-,-,-]]
    ```

