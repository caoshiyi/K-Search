# LayerNorm Tiling<a name="ZH-CN_TOPIC_0000002523303614"></a>

## 功能说明<a name="section618mcpsimp"></a>

Ascend C提供一组LayerNorm Tiling API，方便用户获取LayerNorm kernel计算时所需的Tiling参数。

获取Tiling参数主要分为如下两步：

1.  先通过**GetLayerNormMaxMinTmpSize**获取LayerNorm接口计算所需最大和最小临时空间大小，用于合理分配计算空间。

    kernel侧LayerNorm接口的计算需要开发者预留/申请临时空间，**GetLayerNormMaxMinTmpSize**用于在host侧获取预留/申请的最大最小临时空间大小，开发者基于此范围选择合适的空间大小作为Tiling参数传递到kernel侧使用。

    -   为保证功能正确，预留/申请的临时空间大小不能小于最小临时空间大小；
    -   在最小临时空间-最大临时空间范围内，随着临时空间增大，kernel侧接口计算性能会有一定程度的优化提升。为了达到更好的性能，开发者可以根据实际的内存使用情况进行空间预留/申请。

2.  通过**GetLayerNormNDTilingInfo**获取LayerNorm kernel侧接口所需tiling参数，需要传入输入shape，剩余的可供LayerNorm接口计算的空间大小和计算的数据类型。

    LayerNorm Tiling结构体的定义如下，开发者无需关注该Tiling结构的具体信息，只需要传递到kernel侧，传入LayerNorm高阶API接口，直接进行使用即可。

    -   输出归一化结果、均值和方差的LayerNorm接口所需的Tiling结构体

        ```
        struct LayerNormTiling {
            uint32_t bLength = 0;
            uint32_t sLength = 0;
            uint32_t hLength = 0;
            uint32_t originalHLength = 0;
            uint32_t inputXSize = 0;
            uint32_t meanVarSize = 0;
            uint32_t numberOfTmpBuf = 0;
            uint32_t meanTmpTensorPos = 0;
            uint32_t meanTmpTensorSize = 0;
            uint32_t varianceTmpTensorPos = 0;
            uint32_t varianceTmpTensorSize = 0;
            uint32_t tmpBufSize = 0;
            uint32_t oneTmpSize = 0;
            uint32_t firstTmpStartPos = 0;
            uint32_t secondTmpStartPos = 0;
            uint32_t thirdTmpStartPos = 0;
            uint32_t loopRound = 0;
            uint32_t inputRoundSize = 0;
            uint32_t inputTailSize = 0;
            uint32_t inputTailPos = 0;
            uint32_t meanVarRoundSize = 0;
            uint32_t meanVarTailSize = 0;
            uint32_t meanVarTailPos = 0;
            uint32_t bshCurLength = 0;
            uint32_t bsCurLength = 0;
            float lastDimValueBack = 0.0;
        };
        ```

    -   输出归一化结果、均值和标准差的倒数的LayerNorm接口所需的Tiling结构体

        ```
        struct LayerNormSeparateTiling{
            uint32_t aLength = 0;
            uint32_t rLength = 0;
            uint32_t halfAddRepeatTimes = 0;
            uint32_t rHeadLength = 0;
            float k2Rec = 0;
            float k2RRec = 0;
            uint32_t inputXSize = 0;
            uint32_t meanVarSize = 0;
            uint32_t numberOfTmpBuf = 0;
            uint32_t varianceTmpTensorPos = 0;
            uint32_t varianceTmpTensorSize = 0;
            uint32_t tmpBufSize = 0;
            uint32_t oneTmpSize = 0;
            uint32_t firstTmpStartPos = 0;
            uint32_t secondTmpStartPos = 0;
            uint32_t thirdTmpStartPos = 0;
            uint32_t loopRound = 0;
            uint32_t inputRoundSize = 0;
            uint32_t inputTailSize = 0;
            uint32_t inputTailPos = 0;
            uint32_t meanVarRoundSize = 0;
            uint32_t meanVarTailSize = 0;
            uint32_t meanVarTailPos = 0;
            uint32_t arCurLength = 0;
            uint32_t aCurLength = 0;
            float rValueBack = 0;
        };
        ```

## 函数原型<a name="section620mcpsimp"></a>

> **说明：** 
>GetLayerNormNDTillingInfo接口废弃，并将在后续版本移除，请不要使用该接口。请使用GetLayerNormNDTilingInfo接口。

-   GetLayerNormMaxMinTmpSize接口
    -   输出归一化结果、均值和方差的LayerNorm接口所需的临时空间

        ```
        void GetLayerNormMaxMinTmpSize(const ge::Shape& srcShape, const uint32_t typeSize, const bool isReuseSource, uint32_t& maxValue, uint32_t& minValue)
        ```

    -   输出归一化结果、均值和标准差的倒数的LayerNorm接口所需的临时空间

        ```
        void GetLayerNormMaxMinTmpSize(const ge::Shape& srcShape, const uint32_t typeSize, const bool isReuseSource, const bool isComputeRstd, const bool isOnlyOutput, uint32_t& maxValue, uint32_t& minValue)
        ```

-   GetLayerNormNDTilingInfo/GetLayerNormNDTillingInfo接口
    -   输出归一化结果、均值和方差的LayerNorm接口所需的tiling参数

        ```
        void GetLayerNormNDTilingInfo(const ge::Shape& srcShape, const uint32_t stackBufferSize, const uint32_t typeSize, const bool isReuseSource, optiling::LayerNormTiling& tiling)
        ```

        ```
        void GetLayerNormNDTilingInfo(const ge::Shape& srcShape, const uint32_t stackBufferSize, const uint32_t typeSize, const bool isReuseSource, AscendC::tiling::LayerNormTiling& tiling)
        ```

    -   输出归一化结果、均值和方差的LayerNorm接口所需的tiling参数（不推荐使用）

        ```
        void GetLayerNormNDTillingInfo(const ge::Shape& srcShape, const uint32_t stackBufferSize, const uint32_t typeSize, const bool isReuseSource, optiling::LayerNormTiling& tilling)
        ```

    -   输出归一化结果、均值和标准差的倒数的LayerNorm接口所需的tiling参数

        ```
        void GetLayerNormNDTilingInfo(const ge::Shape& srcShape, const uint32_t stackBufferSize, const uint32_t typeSize, const bool isReuseSource, const bool isComputeRstd, optiling::LayerNormSeparateTiling& tiling)
        ```

        ```
        void GetLayerNormNDTilingInfo(const ge::Shape& srcShape, const uint32_t stackBufferSize, const uint32_t typeSize, const bool isReuseSource, const bool isComputeRstd, AscendC::tiling::LayerNormSeparateTiling& tiling)
        ```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  GetLayerNormMaxMinTmpSize接口参数列表

<a name="table729818506422"></a>
<table><thead align="left"><tr id="row11299950204217"><th class="cellrowborder" valign="top" width="16.89%" id="mcps1.2.4.1.1"><p id="p1029955044218"><a name="p1029955044218"></a><a name="p1029955044218"></a>接口</p>
</th>
<th class="cellrowborder" valign="top" width="9.950000000000001%" id="mcps1.2.4.1.2"><p id="p13216203215467"><a name="p13216203215467"></a><a name="p13216203215467"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="73.16%" id="mcps1.2.4.1.3"><p id="p1629911506421"><a name="p1629911506421"></a><a name="p1629911506421"></a>功能</p>
</th>
</tr>
</thead>
<tbody><tr id="row12299165018421"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p1329915004219"><a name="p1329915004219"></a><a name="p1329915004219"></a>srcShape</p>
</td>
<td class="cellrowborder" valign="top" width="9.950000000000001%" headers="mcps1.2.4.1.2 "><p id="p521618329466"><a name="p521618329466"></a><a name="p521618329466"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.16%" headers="mcps1.2.4.1.3 "><a name="ul45501115191"></a><a name="ul45501115191"></a><ul id="ul45501115191"><li>输出归一化结果、均值和方差的LayerNorm接口：<p id="p8299155010420"><a name="p8299155010420"></a><a name="p8299155010420"></a>输入数据inputX的shape信息{B, S, storageHLength, originHLength}，包括当前输入的inputX的shape信息，以及地址对齐前（如存在H轴补齐操作）的原有shape信息 。</p>
<p id="p7451446105616"><a name="p7451446105616"></a><a name="p7451446105616"></a>在API支持的场景下，storageHLength和originHLength保持一致。</p>
</li></ul>
<a name="ul38072284917"></a><a name="ul38072284917"></a><ul id="ul38072284917"><li>输出归一化结果、均值和标准差的倒数的LayerNorm接口：<p id="p14476194110720"><a name="p14476194110720"></a><a name="p14476194110720"></a>输入数据inputX的shape信息{A, R}，A轴长度可以在kernel接口中动态指定，但范围不能超过此参数中A的大小。</p>
</li></ul>
</td>
</tr>
<tr id="row19299125011422"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p1929916502421"><a name="p1929916502421"></a><a name="p1929916502421"></a>typeSize</p>
</td>
<td class="cellrowborder" valign="top" width="9.950000000000001%" headers="mcps1.2.4.1.2 "><p id="p172166321461"><a name="p172166321461"></a><a name="p172166321461"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.16%" headers="mcps1.2.4.1.3 "><p id="p8299150154218"><a name="p8299150154218"></a><a name="p8299150154218"></a>输入数据inputX的数据类型大小，单位为字节。比如输入的数据类型为half，此处应传入2。</p>
</td>
</tr>
<tr id="row5299125054217"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p9777142884312"><a name="p9777142884312"></a><a name="p9777142884312"></a>isReuseSource</p>
</td>
<td class="cellrowborder" valign="top" width="9.950000000000001%" headers="mcps1.2.4.1.2 "><p id="p1221613214466"><a name="p1221613214466"></a><a name="p1221613214466"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.16%" headers="mcps1.2.4.1.3 "><p id="p277815284439"><a name="p277815284439"></a><a name="p277815284439"></a>是否复用源操作数的内存空间，与<a href="LayerNorm.md">LayerNorm</a>接口一致。</p>
</td>
</tr>
<tr id="row1495114614816"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p17710155515616"><a name="p17710155515616"></a><a name="p17710155515616"></a>isComputeRstd</p>
</td>
<td class="cellrowborder" valign="top" width="9.950000000000001%" headers="mcps1.2.4.1.2 "><p id="p6710175514616"><a name="p6710175514616"></a><a name="p6710175514616"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.16%" headers="mcps1.2.4.1.3 "><p id="p27102557620"><a name="p27102557620"></a><a name="p27102557620"></a>是否计算标准差的倒数rstd。用于Tiling中区分选择的LayerNorm API。</p>
</td>
</tr>
<tr id="row10844285819"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p6710145517617"><a name="p6710145517617"></a><a name="p6710145517617"></a>isOnlyOutput</p>
</td>
<td class="cellrowborder" valign="top" width="9.950000000000001%" headers="mcps1.2.4.1.2 "><p id="p87101555263"><a name="p87101555263"></a><a name="p87101555263"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.16%" headers="mcps1.2.4.1.3 "><p id="p410018335436"><a name="p410018335436"></a><a name="p410018335436"></a>是否只输出y，不输出均值mean与标准差的倒数rstd。当前该参数仅支持false，y、mean和rstd的结果全都输出。</p>
</td>
</tr>
<tr id="row6777152811436"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p23791451102416"><a name="p23791451102416"></a><a name="p23791451102416"></a>maxValue</p>
</td>
<td class="cellrowborder" valign="top" width="9.950000000000001%" headers="mcps1.2.4.1.2 "><p id="p470071112510"><a name="p470071112510"></a><a name="p470071112510"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="73.16%" headers="mcps1.2.4.1.3 "><p id="p1288155462517"><a name="p1288155462517"></a><a name="p1288155462517"></a>输出LayerNorm接口所需的tiling信息（最大临时空间大小）。</p>
<p id="p1078374516308"><a name="p1078374516308"></a><a name="p1078374516308"></a>LayerNorm接口能完成计算所需的最大临时空间大小，超出该值的空间不会被该接口使用。在最小临时空间-最大临时空间范围内，随着临时空间增大，kernel侧接口计算性能会有一定程度的优化提升。为了达到更好的性能，开发者可以根据实际的内存使用情况进行空间预留/申请。</p>
<div class="note" id="note1275197121212"><a name="note1275197121212"></a><a name="note1275197121212"></a><span class="notetitle"> 说明： </span><div class="notebody"><p id="p1040915016493"><a name="p1040915016493"></a><a name="p1040915016493"></a>maxValue仅作为参考值，有可能大于<span id="ph1088254310583"><a name="ph1088254310583"></a><a name="ph1088254310583"></a>Unified Buffer</span>剩余空间的大小，该场景下，开发者需要根据<span id="ph20796650203718"><a name="ph20796650203718"></a><a name="ph20796650203718"></a>Unified Buffer</span>剩余空间的大小来选取合适的临时空间大小。</p>
</div></div>
</td>
</tr>
<tr id="row6563634154317"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p956314345431"><a name="p956314345431"></a><a name="p956314345431"></a>minValue</p>
</td>
<td class="cellrowborder" valign="top" width="9.950000000000001%" headers="mcps1.2.4.1.2 "><p id="p42161432144610"><a name="p42161432144610"></a><a name="p42161432144610"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="73.16%" headers="mcps1.2.4.1.3 "><p id="p20807932162517"><a name="p20807932162517"></a><a name="p20807932162517"></a>输出LayerNorm接口所需的tiling信息（最小临时空间大小）。</p>
<p id="p281475823012"><a name="p281475823012"></a><a name="p281475823012"></a>LayerNorm接口能完成计算所需最小临时空间大小。为保证功能正确，接口计算时预留/申请的临时空间不能小于该数值。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  GetLayerNormNDTilingInfo和GetLayerNormNDTillingInfo接口参数列表

<a name="table667463712434"></a>
<table><thead align="left"><tr id="row5674103712436"><th class="cellrowborder" valign="top" width="19.63196319631963%" id="mcps1.2.4.1.1"><p id="p16674193714437"><a name="p16674193714437"></a><a name="p16674193714437"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="10.771077107710772%" id="mcps1.2.4.1.2"><p id="p8674133764310"><a name="p8674133764310"></a><a name="p8674133764310"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="69.5969596959696%" id="mcps1.2.4.1.3"><p id="p1367413376436"><a name="p1367413376436"></a><a name="p1367413376436"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row15674237164319"><td class="cellrowborder" valign="top" width="19.63196319631963%" headers="mcps1.2.4.1.1 "><p id="p1467483715439"><a name="p1467483715439"></a><a name="p1467483715439"></a>srcShape</p>
</td>
<td class="cellrowborder" valign="top" width="10.771077107710772%" headers="mcps1.2.4.1.2 "><p id="p267433711433"><a name="p267433711433"></a><a name="p267433711433"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="69.5969596959696%" headers="mcps1.2.4.1.3 "><a name="ul16991433181119"></a><a name="ul16991433181119"></a><ul id="ul16991433181119"><li>输出归一化结果、均值和方差的LayerNorm接口：<p id="p46835414433"><a name="p46835414433"></a><a name="p46835414433"></a>输入数据inputX的shape信息{B, S, storageHLength, originHLength}，包括当前输入的inputX的shape信息，以及地址对齐前（如存在H轴补齐操作）的原有shape信息。</p>
</li></ul>
<a name="ul10342103816111"></a><a name="ul10342103816111"></a><ul id="ul10342103816111"><li>输出归一化结果、均值和标准差的倒数的LayerNorm接口：<p id="p135274241112"><a name="p135274241112"></a><a name="p135274241112"></a>输入数据inputX的shape信息{A, R}，A轴长度可以在kernel接口中动态指定，但范围不能超过此参数中A的大小。</p>
</li></ul>
</td>
</tr>
<tr id="row19723118144412"><td class="cellrowborder" valign="top" width="19.63196319631963%" headers="mcps1.2.4.1.1 "><p id="p4723188154413"><a name="p4723188154413"></a><a name="p4723188154413"></a>stackBufferSize</p>
</td>
<td class="cellrowborder" valign="top" width="10.771077107710772%" headers="mcps1.2.4.1.2 "><p id="p15723148144416"><a name="p15723148144416"></a><a name="p15723148144416"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="69.5969596959696%" headers="mcps1.2.4.1.3 "><p id="p77236817446"><a name="p77236817446"></a><a name="p77236817446"></a>可供LayerNorm接口使用的空间大小，单位Byte。</p>
</td>
</tr>
<tr id="row44731452154416"><td class="cellrowborder" valign="top" width="19.63196319631963%" headers="mcps1.2.4.1.1 "><p id="p15473135218445"><a name="p15473135218445"></a><a name="p15473135218445"></a>typeSize</p>
</td>
<td class="cellrowborder" valign="top" width="10.771077107710772%" headers="mcps1.2.4.1.2 "><p id="p144731152144416"><a name="p144731152144416"></a><a name="p144731152144416"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="69.5969596959696%" headers="mcps1.2.4.1.3 "><p id="p597311610461"><a name="p597311610461"></a><a name="p597311610461"></a>输入的数据类型大小，单位为字节。比如输入的数据类型为half，此处应传入2。</p>
</td>
</tr>
<tr id="row1017792113457"><td class="cellrowborder" valign="top" width="19.63196319631963%" headers="mcps1.2.4.1.1 "><p id="p8178321144519"><a name="p8178321144519"></a><a name="p8178321144519"></a>isReuseSource</p>
</td>
<td class="cellrowborder" valign="top" width="10.771077107710772%" headers="mcps1.2.4.1.2 "><p id="p16178152124518"><a name="p16178152124518"></a><a name="p16178152124518"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="69.5969596959696%" headers="mcps1.2.4.1.3 "><p id="p18178172110459"><a name="p18178172110459"></a><a name="p18178172110459"></a>是否可以复用inputX的内存空间。</p>
</td>
</tr>
<tr id="row22171206138"><td class="cellrowborder" valign="top" width="19.63196319631963%" headers="mcps1.2.4.1.1 "><p id="p1462812816177"><a name="p1462812816177"></a><a name="p1462812816177"></a>isComputeRstd</p>
</td>
<td class="cellrowborder" valign="top" width="10.771077107710772%" headers="mcps1.2.4.1.2 "><p id="p862882811711"><a name="p862882811711"></a><a name="p862882811711"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="69.5969596959696%" headers="mcps1.2.4.1.3 "><p id="p080755112188"><a name="p080755112188"></a><a name="p080755112188"></a>是否计算标准差的倒数rstd。用于Tiling中区分选择的LayerNorm API。</p>
</td>
</tr>
<tr id="row8764163917458"><td class="cellrowborder" valign="top" width="19.63196319631963%" headers="mcps1.2.4.1.1 "><p id="p376410391458"><a name="p376410391458"></a><a name="p376410391458"></a>tilling</p>
</td>
<td class="cellrowborder" valign="top" width="10.771077107710772%" headers="mcps1.2.4.1.2 "><p id="p576417394455"><a name="p576417394455"></a><a name="p576417394455"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="69.5969596959696%" headers="mcps1.2.4.1.3 "><p id="p0764153917454"><a name="p0764153917454"></a><a name="p0764153917454"></a>输入数据的切分信息。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section92611953111217"></a>

无

## 调用示例<a name="section94691236101419"></a>

如下样例介绍了使用输出方差的LayerNorm高阶API时，host侧获取Tiling参数的流程以及该参数如何在kernel侧使用。样例中输入Tensor的shape大小为\[2, 16, 64\]，输入的数据类型为half。

1.  将LayerNormTiling结构体参数增加至TilingData结构体，作为TilingData结构体的一个字段。

    ```
    BEGIN_TILING_DATA_DEF(TilingData)               // 注册一个tiling的类，以tiling的名字作为入参
      TILING_DATA_FIELD_DEF(uint32_t, totalLength); // 添加tiling字段，总计算数据量
      TILING_DATA_FIELD_DEF(uint32_t, tileNum);     // 添加tiling字段，每个核上总计算数据分块个数
      ...                                           // 添加其他tiling字段
      TILING_DATA_FIELD_DEF_STRUCT(LayerNormTiling, layernormTilingData); // 将LayerNormTiling结构体参数增加至TilingData结构体
    END_TILING_DATA_DEF;
    ```

2.  Tiling实现函数中，首先调用GetLayerNormMaxMinTmpSize接口获取LayerNorm接口能完成计算所需最大/最小临时空间大小，根据该范围结合实际的内存使用情况设置合适的空间大小，然后调用GetLayerNormNDTilingInfo接口根据输入shape、剩余的可供计算的空间大小等信息获取LayerNorm kernel侧接口所需tiling参数。

    ```
    namespace optiling {
    const uint32_t NUM_BLOCKS = 8;
    const uint32_t TILE_NUM = 8;
    static ge::graphStatus TilingFunc(gert::TilingContext* context)
    {
        TilingData tiling;
        uint32_t totalLength = context->GetInputTensor(0)->GetShapeSize();
        context->SetBlockDim(NUM_BLOCKS);
        tiling.set_totalLength(totalLength);
        tiling.set_tileNum(TILE_NUM);
        // 设置其他Tiling参数
        ...
        // {B, S, storageHLength, originHLength}
        std::vector<int64_t> shapeVec = {2, 16, 64, 64};
        ge::Shape srcShape(shapeVec);
        // 本样例中仅作为样例说明，通过GetLayerNormMaxMinTmpSize获取最小值并传入，来保证功能正确，开发者可以根据需要传入合适的空间大小
        uint32_t max;
        uint32_t min;
        AscendC::GetLayerNormMaxMinTmpSize(srcShape, sizeof(half), false, max, min);
        // 获取Layernorm Tiling参数
        AscendC::GetLayerNormNDTilingInfo(srcShape, min, sizeof(half), false, tiling.layernormTilingData); 
         ... // 其他逻辑
        tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
        context->SetTilingKey(1);
        return ge::GRAPH_SUCCESS;
    }
    } // namespace optiling
    ```

3.  对应的kernel侧通过在核函数中调用GET\_TILING\_DATA获取TilingData，继而将TilingData中的LayerNormTiling信息传入LayerNorm接口参与计算。完整的kernel侧样例请参考[LayerNorm](LayerNorm.md#section7848153554510)。

    ```
    extern "C" __global__ __aicore__ void func_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
    {
        GET_TILING_DATA(tilingData, tiling);
        KernelFunc op;
        op.Init(x, y, z, tilingData.totalLength, tilingData.tileNum,tilingData.layernormTilingData);
        if (TILING_KEY_IS(1)) {
            op.Process();
        }
    }
    ```

如下样例介绍了使用输出标准差的倒数的LayerNorm高阶API时，host侧获取Tiling参数的流程以及该参数如何在kernel侧使用。样例中输入Tensor的shape大小为\[2, 64\]，输入的数据类型为half。

1.  将LayerNormTiling结构体参数增加至TilingData结构体，作为TilingData结构体的一个字段。

    ```
    BEGIN_TILING_DATA_DEF(TilingData)                         // 注册一个tiling的类，以tiling的名字作为入参
      TILING_DATA_FIELD_DEF(uint32_t, aLength);                // 添加tiling字段，a轴长度
      TILING_DATA_FIELD_DEF(uint32_t, rLengthWithPadding);     // 添加tiling字段，r轴对齐32B后的长度
      ...                                                     // 添加其他tiling字段
      TILING_DATA_FIELD_DEF_STRUCT(LayerNormSeparateTiling, layernormTilingData); // 将LayerNormSeparateTiling结构体参数增加至TilingData结构体
    END_TILING_DATA_DEF;
    ```

2.  Tiling实现函数中，首先调用GetLayerNormMaxMinTmpSize接口获取LayerNorm接口能完成计算所需最大/最小临时空间大小，根据该范围结合实际的内存使用情况设置合适的空间大小，然后调用GetLayerNormNDTilingInfo接口根据输入shape、剩余的可供计算的空间大小等信息获取LayerNorm kernel侧接口所需tiling参数。

    ```
    namespace optiling {
    const uint32_t NUM_BLOCKS = 1;
    const uint32_t TILE_NUM = 8;
    static ge::graphStatus TilingFunc(gert::TilingContext* context)
    {
        TilingData tiling;
        uint32_t totalLength = context->GetInputTensor(0)->GetShapeSize();
        context->SetBlockDim(NUM_BLOCKS);
        tiling.set_totalLength(totalLength);
        tiling.set_tileNum(TILE_NUM);
        // 设置其他Tiling参数
        ...
        // {A, R}
        std::vector<int64_t> shapeVec = {2, 64};
        ge::Shape srcShape(shapeVec);
        // 本样例中仅作为样例说明，通过GetLayerNormMaxMinTmpSize获取最小值并传入，来保证功能正确，开发者可以根据需要传入合适的空间大小
        uint32_t max;
        uint32_t min;
        AscendC::GetLayerNormMaxMinTmpSize(srcShape, sizeof(half), false, true, false, max, min);
        // 获取Layernorm Tiling参数
        AscendC::GetLayerNormNDTilingInfo(srcShape, min, sizeof(half), false, true, tiling.layernormTilingData); 
        // auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        // AscendC::GetLayerNormMaxMinTmpSize(srcShape, sizeof(half), false, true, false, ascendcPlatform, max, min);
        // 获取Layernorm Tiling参数
        // AscendC::GetLayerNormNDTilingInfo(srcShape, min, sizeof(half), false, true, ascendcPlatform, tiling.layernormTilingData);
         ... // 其他逻辑
        tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
        context->SetTilingKey(1);
        return ge::GRAPH_SUCCESS;
    }
    } // namespace optiling
    ```

3.  对应的kernel侧通过在核函数中调用GET\_TILING\_DATA获取TilingData，继而将TilingData中的LayerNormTiling信息传入LayerNorm接口参与计算。完整的kernel侧样例请参考[LayerNorm](LayerNorm.md#section7848153554510)。

    ```
    extern "C" __global__ __aicore__ void func_custom(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR mean, GM_ADDR rstd, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
    {
        GET_TILING_DATA(tilingData, tiling);
        float epsilon = tilingData.epsilon;
        AscendC::LayerNormPara para(tilingData.aLength, tilingData.rLengthWithPadding);
        KernelFunc op;
        op.Init(x, gamma, beta, mean, rstd, y, epsilon, para, tilingData.layernormTilingData);
        if (TILING_KEY_IS(1)) {
            op.Process();
        }
    }
    ```

