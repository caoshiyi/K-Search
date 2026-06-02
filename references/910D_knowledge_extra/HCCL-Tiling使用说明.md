# HCCL Tiling使用说明<a name="ZH-CN_TOPIC_0000002523343966"></a>

> **说明：** 
>对于[TilingData结构体](TilingData结构体.md)，在定义通算融合算子的Tiling结构体时，[Mc2InitTiling](TilingData结构体.md#table4835205712588)必须定义为算子Tiling结构体的第一个参数，[Mc2CcTiling](TilingData结构体.md#table678914014562)对于在算子Tiling结构体中被定义的位置没有要求。

根据[使用标准C++语法定义Tiling结构体](使用标准C++语法定义Tiling结构体.md)的方式，Ascend C提供一组HCCL Tiling API，方便用户获取HCCL Kernel计算时所需的Tiling参数。您只需要传入通信的相关信息，调用API接口，即可获取通信相关的Tiling参数。

HCCL Tiling API获取Tiling参数的流程如下：

1.  创建一个[Mc2CcTilingConfig](HCCL-Tiling构造函数.md)类对象。

    ```
    const char *groupName = "testGroup";
    uint32_t opType = HCCL_CMD_REDUCE_SCATTER;
    std::string algConfig = "ReduceScatter=level0:fullmesh";
    uint32_t reduceType = HCCL_REDUCE_SUM;
    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(groupName, opType, algConfig, reduceType);
    ```

2.  通过配置接口设置通信信息（可选）。

    ```
    mc2CcTilingConfig.SetSkipLocalRankCopy(0);
    mc2CcTilingConfig.SetSkipBufferWindowCopy(1);
    ```

    可调用的配置接口列于下表。

    **表 1**  Mc2CcTilingConfig类对象的配置接口列表

    <a name="table282771234319"></a>
    <table><thead align="left"><tr id="row1982711244317"><th class="cellrowborder" valign="top" width="30.75%" id="mcps1.2.3.1.1"><p id="p6827201224320"><a name="p6827201224320"></a><a name="p6827201224320"></a>接口</p>
    </th>
    <th class="cellrowborder" valign="top" width="69.25%" id="mcps1.2.3.1.2"><p id="p1182716120437"><a name="p1182716120437"></a><a name="p1182716120437"></a>功能</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="row1782771218436"><td class="cellrowborder" valign="top" width="30.75%" headers="mcps1.2.3.1.1 "><p id="p8827181212432"><a name="p8827181212432"></a><a name="p8827181212432"></a><a href="SetOpType.md">SetOpType</a></p>
    </td>
    <td class="cellrowborder" valign="top" width="69.25%" headers="mcps1.2.3.1.2 "><p id="p15827101214433"><a name="p15827101214433"></a><a name="p15827101214433"></a>设置通信任务类型。</p>
    </td>
    </tr>
    <tr id="row782731215433"><td class="cellrowborder" valign="top" width="30.75%" headers="mcps1.2.3.1.1 "><p id="p6827151264316"><a name="p6827151264316"></a><a name="p6827151264316"></a><a href="SetGroupName.md">SetGroupName</a></p>
    </td>
    <td class="cellrowborder" valign="top" width="69.25%" headers="mcps1.2.3.1.2 "><p id="p16827912164316"><a name="p16827912164316"></a><a name="p16827912164316"></a>设置通信任务所在的通信域。</p>
    </td>
    </tr>
    <tr id="row88271812174310"><td class="cellrowborder" valign="top" width="30.75%" headers="mcps1.2.3.1.1 "><p id="p182781212436"><a name="p182781212436"></a><a name="p182781212436"></a><a href="SetAlgConfig.md">SetAlgConfig</a></p>
    </td>
    <td class="cellrowborder" valign="top" width="69.25%" headers="mcps1.2.3.1.2 "><p id="p1082811294319"><a name="p1082811294319"></a><a name="p1082811294319"></a>设置通信算法。</p>
    </td>
    </tr>
    <tr id="row14828161264317"><td class="cellrowborder" valign="top" width="30.75%" headers="mcps1.2.3.1.1 "><p id="p382851217436"><a name="p382851217436"></a><a name="p382851217436"></a><a href="SetReduceType.md">SetReduceType</a></p>
    </td>
    <td class="cellrowborder" valign="top" width="69.25%" headers="mcps1.2.3.1.2 "><p id="p148281612194318"><a name="p148281612194318"></a><a name="p148281612194318"></a>设置Reduce操作类型。</p>
    </td>
    </tr>
    <tr id="row882881217431"><td class="cellrowborder" valign="top" width="30.75%" headers="mcps1.2.3.1.1 "><p id="p10828512104312"><a name="p10828512104312"></a><a name="p10828512104312"></a><a href="SetStepSize.md">SetStepSize</a></p>
    </td>
    <td class="cellrowborder" valign="top" width="69.25%" headers="mcps1.2.3.1.2 "><p id="p118288126439"><a name="p118288126439"></a><a name="p118288126439"></a>设置细粒度通信时，通信算法的步长。</p>
    </td>
    </tr>
    <tr id="row782831214320"><td class="cellrowborder" valign="top" width="30.75%" headers="mcps1.2.3.1.1 "><p id="p1082821234318"><a name="p1082821234318"></a><a name="p1082821234318"></a><a href="SetSkipLocalRankCopy.md">SetSkipLocalRankCopy</a></p>
    </td>
    <td class="cellrowborder" valign="top" width="69.25%" headers="mcps1.2.3.1.2 "><p id="p182881264313"><a name="p182881264313"></a><a name="p182881264313"></a>设置本卡的通信算法的计算结果是否输出到recvBuf。</p>
    </td>
    </tr>
    <tr id="row118281312174319"><td class="cellrowborder" valign="top" width="30.75%" headers="mcps1.2.3.1.1 "><p id="p1682831219434"><a name="p1682831219434"></a><a name="p1682831219434"></a><a href="SetSkipBufferWindowCopy.md">SetSkipBufferWindowCopy</a></p>
    </td>
    <td class="cellrowborder" valign="top" width="69.25%" headers="mcps1.2.3.1.2 "><p id="p982871224317"><a name="p982871224317"></a><a name="p982871224317"></a>设置通信算法获取输入数据的位置。</p>
    </td>
    </tr>
    <tr id="row1082815124438"><td class="cellrowborder" valign="top" width="30.75%" headers="mcps1.2.3.1.1 "><p id="p582861220437"><a name="p582861220437"></a><a name="p582861220437"></a><a href="SetDebugMode.md">SetDebugMode</a></p>
    </td>
    <td class="cellrowborder" valign="top" width="69.25%" headers="mcps1.2.3.1.2 "><p id="p198281312144311"><a name="p198281312144311"></a><a name="p198281312144311"></a>设置调测模式。</p>
    </td>
    </tr>
    </tbody>
    </table>

3.  调用[GetTiling](GetTiling-121.md)接口，获取Tiling信息。

    ```
    mc2CcTilingConfig.GetTiling(tiling->mc2InitTiling); // tiling为算子组装的TilingData结构体
    mc2CcTilingConfig.GetTiling(tiling->reduceScatterTiling);
    ```

