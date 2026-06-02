# Init<a name="ZH-CN_TOPIC_0000002554424719"></a>

## 产品支持情况<a name="section1550532418810"></a>

<a name="table1334714391211"></a>
<table><thead align="left"><tr id="row1334743121213"><th class="cellrowborder" valign="top" width="53.64%" id="mcps1.1.4.1.1"><p id="p834713321216"><a name="p834713321216"></a><a name="p834713321216"></a><span id="ph834783101215"><a name="ph834783101215"></a><a name="ph834783101215"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="22.59%" id="mcps1.1.4.1.2"><p id="p2347234127"><a name="p2347234127"></a><a name="p2347234127"></a>Tiling参数传入栈地址的接口</p>
</th>
<th class="cellrowborder" align="center" valign="top" width="23.77%" id="mcps1.1.4.1.3"><p id="p126561621806"><a name="p126561621806"></a><a name="p126561621806"></a>Tiling参数传入GM地址的接口</p>
</th>
</tr>
</thead>
<tbody><tr id="row113472312122"><td class="cellrowborder" valign="top" width="53.64%" headers="mcps1.1.4.1.1 "><p id="p234710320128"><a name="p234710320128"></a><a name="p234710320128"></a><span id="ph103471336127"><a name="ph103471336127"></a><a name="ph103471336127"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="22.59%" headers="mcps1.1.4.1.2 "><p id="p4751940181211"><a name="p4751940181211"></a><a name="p4751940181211"></a>√</p>
</td>
<td class="cellrowborder" align="center" valign="top" width="23.77%" headers="mcps1.1.4.1.3 "><p id="p136569216013"><a name="p136569216013"></a><a name="p136569216013"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section618mcpsimp"></a>

Init主要用于对Matmul对象中的Tiling数据进行初始化，根据Tiling参数进行资源划分，Tiling参数的具体介绍请参考[Matmul Tiling侧接口](Matmul-Tiling侧接口.md)。

开发者可以先通过[REGIST\_MATMUL\_OBJ](REGIST_MATMUL_OBJ.md)不传入Tiling参数对单个Matmul对象进行初始化，后续通过Init接口单独传入Tiling参数，对Matmul对象中的Tiling数据进行调整。比如，Tiling参数可变的场景下，可以通过多次调用Init来重新设置Tiling参数。

不需要Tiling变更的场景下，推荐使用[REGIST\_MATMUL\_OBJ](REGIST_MATMUL_OBJ.md)传入Tiling参数进行初始化。

## 函数原型<a name="section620mcpsimp"></a>

-   Tiling参数传入栈地址

    ```
    __aicore__ inline void Init(const TCubeTiling* __restrict cubeTiling, TPipe* tpipe = nullptr)
    ```

-   Tiling参数传入GM地址

    ```
    __aicore__ inline void Init(const __gm__ TCubeTiling* gmCubeTiling, TPipe* tpipe = nullptr)
    ```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  Tiling参数传入栈地址接口参数说明

<a name="zh-cn_topic_0235751031_table33761356"></a>
<table><thead align="left"><tr id="zh-cn_topic_0235751031_row27598891"><th class="cellrowborder" valign="top" width="17.77%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="11.799999999999999%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="70.43%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row42461942101815"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p163481714145518"><a name="p163481714145518"></a><a name="p163481714145518"></a>cubeTiling</p>
</td>
<td class="cellrowborder" valign="top" width="11.799999999999999%" headers="mcps1.2.4.1.2 "><p id="p33487148556"><a name="p33487148556"></a><a name="p33487148556"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="70.43%" headers="mcps1.2.4.1.3 "><p id="p13481914175514"><a name="p13481914175514"></a><a name="p13481914175514"></a>Matmul Tiling参数，TCubeTiling结构体定义请参见<a href="TCubeTiling结构体.md#table1563162142915">表1 TCubeTiling结构说明</a>。</p>
<p id="p246271553317"><a name="p246271553317"></a><a name="p246271553317"></a>Tiling参数可以通过host侧<a href="GetTiling.md">GetTiling</a>接口获取，并传递到kernel侧使用。在kernel侧调用<a href="GET_TILING_DATA.md">GET_TILING_DATA</a>实现将Tiling参数搬运到AI Core内的栈空间中，本接口传入Tiling参数中TCubeTiling结构体的栈地址。</p>
</td>
</tr>
<tr id="row1282014916166"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p14807165981614"><a name="p14807165981614"></a><a name="p14807165981614"></a>tpipe</p>
</td>
<td class="cellrowborder" valign="top" width="11.799999999999999%" headers="mcps1.2.4.1.2 "><p id="p0807115911613"><a name="p0807115911613"></a><a name="p0807115911613"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="70.43%" headers="mcps1.2.4.1.3 "><p id="p1380719591162"><a name="p1380719591162"></a><a name="p1380719591162"></a>Tpipe对象。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  Tiling参数传入GM地址接口参数说明

<a name="table12369115811594"></a>
<table><thead align="left"><tr id="row16369658165914"><th class="cellrowborder" valign="top" width="17.77%" id="mcps1.2.4.1.1"><p id="p12369358165913"><a name="p12369358165913"></a><a name="p12369358165913"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="11.799999999999999%" id="mcps1.2.4.1.2"><p id="p2369115825911"><a name="p2369115825911"></a><a name="p2369115825911"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="70.43%" id="mcps1.2.4.1.3"><p id="p9369958135914"><a name="p9369958135914"></a><a name="p9369958135914"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row12369115825919"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p173696581596"><a name="p173696581596"></a><a name="p173696581596"></a>gmCubeTiling</p>
</td>
<td class="cellrowborder" valign="top" width="11.799999999999999%" headers="mcps1.2.4.1.2 "><p id="p336920583592"><a name="p336920583592"></a><a name="p336920583592"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="70.43%" headers="mcps1.2.4.1.3 "><p id="p12369358125920"><a name="p12369358125920"></a><a name="p12369358125920"></a>Matmul Tiling参数，该参数指向gm上的一块内存地址，其中的数据类型是TCubeTiling结构体，TCubeTiling结构体定义请参见<a href="TCubeTiling结构体.md#table1563162142915">表1 TCubeTiling结构说明</a>。</p>
<p id="p1012232793914"><a name="p1012232793914"></a><a name="p1012232793914"></a>Tiling参数可以通过host侧<a href="GetTiling.md">GetTiling</a>接口获取，并传递到kernel侧使用。在kernel侧调用<a href="GET_TILING_DATA_PTR_WITH_STRUCT.md">GET_TILING_DATA_PTR_WITH_STRUCT</a>获取gm上Tiling参数的指针，本接口传入Tiling参数中TCubeTiling结构体的GM地址。</p>
</td>
</tr>
<tr id="row1636955817596"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p183703589598"><a name="p183703589598"></a><a name="p183703589598"></a>tpipe</p>
</td>
<td class="cellrowborder" valign="top" width="11.799999999999999%" headers="mcps1.2.4.1.2 "><p id="p7370185814595"><a name="p7370185814595"></a><a name="p7370185814595"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="70.43%" headers="mcps1.2.4.1.3 "><p id="p1537075820597"><a name="p1537075820597"></a><a name="p1537075820597"></a>Tpipe对象。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

-   Tiling参数传入栈地址的接口：

    无。

-   Tiling参数传入GM地址的接口：
    -   仅支持Matmul Tiling参数的[部分常量化](GetMatmulApiTiling.md#section618mcpsimp)场景。
    -   不支持CPU域调试。

## 调用示例<a name="section1665082013318"></a>

-   Tiling参数传入栈地址

    ```
    GET_TILING_DATA(tilingData, tiling);
    // ...
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm);
    mm.Init(&(tiling.cubeTilingData));
    ```

-   Tiling参数传入GM地址
    -   纯Cube模式

        ```
        #define ASCENDC_CUBE_ONLY
        
        GET_TILING_DATA_PTR_WITH_STRUCT(MatmulCustomTilingData, tilingDataPtr, tiling);
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
        // ...
        REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm);
        mm.Init(&(tilingDataPtr->cubeTilingData));
        ```

    -   MIX模式

        ```
        GET_TILING_DATA_PTR_WITH_STRUCT(MatmulCustomTilingData, tilingDataPtr, tiling);
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
        // ...
        // MIX模式下，只调用REGIST_MATMUL_OBJ接口，传入Tiling参数的GM地址，不需调用Init接口
        REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &(tilingDataPtr->cubeTilingData));
        ```

