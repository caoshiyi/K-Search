# BatchNorm Tiling<a name="ZH-CN_TOPIC_0000002523303746"></a>

## 功能说明<a name="section663724118466"></a>

BatchNorm Tiling API用于获取BatchNorm kernel计算时所需的Tiling参数。获取Tiling参数主要分为如下两步：

1.  通过**GetBatchNormMaxMinTmpSize**获取BatchNorm接口计算所需最大和最小临时空间大小。

    kernel侧BatchNorm接口的计算需要开发者预留/申请临时空间，**GetBatchNormMaxMinTmpSize**用于在host侧获取预留/申请的最大最小临时空间大小，开发者基于此范围选择合适的空间大小作为Tiling参数传递到kernel侧使用。

    -   为保证功能正确，预留/申请的临时空间大小不能小于最小临时空间大小；
    -   在最小临时空间-最大临时空间范围内，随着临时空间增大，kernel侧接口计算性能会有一定程度的优化提升。为了达到更好的性能，开发者可以根据实际的内存使用情况进行空间预留/申请。

2.  通过**GetBatchNormNDTilingInfo**获取BatchNorm kernel侧接口所需tiling参数。

    BatchNorm Tiling结构体的定义如下，开发者无需关注该tiling结构的具体信息，只需要传递到kernel侧，传入BatchNorm高阶API接口，直接进行使用即可。

    ```
    struct BatchNormTiling {
        uint32_t originalBLength = 0;
        uint32_t meanVarSize = 0;
        uint32_t meanTmpTensorPos = 0;
        uint32_t varianceTmpTensorPos = 0;
        uint32_t tmpBufSize = 0;
        uint32_t oneTmpSize = 0;
        uint32_t firstTmpStartPos = 0;
        uint32_t secondTmpStartPos = 0;
        uint32_t thirdTmpStartPos = 0;
        uint32_t loopRound = 0;
        uint32_t inputTailSize = 0;
        uint32_t inputTailPos = 0;
        uint32_t meanVarTailSize = 0;
        uint32_t meanVarTailPos = 0;
        uint32_t bshCurLength = 0;
        uint32_t shCurLength = 0;
        float firstDimValueBack = 0;
        uint32_t castHalfRepStride = 0;
        uint32_t shCurLengthBlockNum = 0;
        uint32_t castHalfOutRepStride = 0;
    };
    ```

## 函数原型<a name="section7471740471"></a>

```
bool GetBatchNormMaxMinTmpSize(const ge::Shape& srcShape, const ge::Shape& originSrcShape, const uint32_t typeSize, const bool isReuseSource, uint32_t& maxValue,uint32_t& minValue, const bool isBasicBlock = false)
```

```
bool GetBatchNormNDTilingInfo(const ge::Shape& srcShape, const ge::Shape& originSrcShape, const uint32_t stackBufferByteSize, const uint32_t typeSize, const bool isReuseSource, optiling::BatchNormTiling& tilling, const bool isBasicBlock = false)
```

```
bool GetBatchNormNDTilingInfo(const ge::Shape& srcShape, const ge::Shape& originSrcShape, const uint32_t stackBufferByteSize, const uint32_t typeSize, const bool isReuseSource, AscendC::tiling::BatchNormTiling& tilling, const bool isBasicBlock = false)
```

## 参数说明<a name="section193991541162015"></a>

**表 1**  GetBatchNormMaxMinTmpSize接口参数说明

<a name="zh-cn_topic_0235751031_table33761356"></a>
<table><thead align="left"><tr id="zh-cn_topic_0235751031_row27598891"><th class="cellrowborder" valign="top" width="17.77%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="9.69%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="72.54%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row42461942101815"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p189045212538"><a name="p189045212538"></a><a name="p189045212538"></a>srcShape</p>
</td>
<td class="cellrowborder" valign="top" width="9.69%" headers="mcps1.2.4.1.2 "><p id="p490205216537"><a name="p490205216537"></a><a name="p490205216537"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.54%" headers="mcps1.2.4.1.3 "><p id="p1897893119610"><a name="p1897893119610"></a><a name="p1897893119610"></a>输入数据inputX的shape信息[B, S, H]，S*H需要32B对齐。</p>
</td>
</tr>
<tr id="row19230246194511"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p923164634515"><a name="p923164634515"></a><a name="p923164634515"></a>originSrcShape</p>
</td>
<td class="cellrowborder" valign="top" width="9.69%" headers="mcps1.2.4.1.2 "><p id="p82314469451"><a name="p82314469451"></a><a name="p82314469451"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.54%" headers="mcps1.2.4.1.3 "><p id="p142311046104512"><a name="p142311046104512"></a><a name="p142311046104512"></a>输入数据inputX的originshape信息[originB, originS, originH]。</p>
</td>
</tr>
<tr id="row250163914199"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p690155215538"><a name="p690155215538"></a><a name="p690155215538"></a>typeSize</p>
</td>
<td class="cellrowborder" valign="top" width="9.69%" headers="mcps1.2.4.1.2 "><p id="p790195215536"><a name="p790195215536"></a><a name="p790195215536"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.54%" headers="mcps1.2.4.1.3 "><p id="p8299150154218"><a name="p8299150154218"></a><a name="p8299150154218"></a>输入的数据类型大小，单位为字节。比如输入的数据类型为half，此处应传入2。</p>
</td>
</tr>
<tr id="row041413392198"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p6901552165311"><a name="p6901552165311"></a><a name="p6901552165311"></a>isReuseSource</p>
</td>
<td class="cellrowborder" valign="top" width="9.69%" headers="mcps1.2.4.1.2 "><p id="p390135217539"><a name="p390135217539"></a><a name="p390135217539"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.54%" headers="mcps1.2.4.1.3 "><p id="p14902526537"><a name="p14902526537"></a><a name="p14902526537"></a>中间变量是否能够复用输入内存。该参数预留，传入默认值false即可。</p>
</td>
</tr>
<tr id="row17371444131520"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p119016528533"><a name="p119016528533"></a><a name="p119016528533"></a>maxValue</p>
</td>
<td class="cellrowborder" valign="top" width="9.69%" headers="mcps1.2.4.1.2 "><p id="p590155210531"><a name="p590155210531"></a><a name="p590155210531"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="72.54%" headers="mcps1.2.4.1.3 "><p id="p1890195216531"><a name="p1890195216531"></a><a name="p1890195216531"></a>BatchNorm接口能完成计算所需的最大临时空间大小，超出max的空间不会被该接口使用。在min-max范围内，预留/申请空间越大，接口计算性能越好。为了达到更好的性能，开发者可以根据实际的内存使用情况进行空间预留/申请。maxValue为0表示计算不需要临时空间。</p>
<div class="note" id="note1275197121212"><a name="note1275197121212"></a><a name="note1275197121212"></a><span class="notetitle"> 说明： </span><div class="notebody"><p id="p1040915016493"><a name="p1040915016493"></a><a name="p1040915016493"></a>maxValue仅作为参考值，有可能大于<span id="ph1088254310583"><a name="ph1088254310583"></a><a name="ph1088254310583"></a>Unified Buffer</span>剩余空间的大小，该场景下，开发者需要根据<span id="ph20796650203718"><a name="ph20796650203718"></a><a name="ph20796650203718"></a>Unified Buffer</span>剩余空间的大小来选取合适的临时空间大小。</p>
</div></div>
</td>
</tr>
<tr id="row1082704235314"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p169085211534"><a name="p169085211534"></a><a name="p169085211534"></a>minValue</p>
</td>
<td class="cellrowborder" valign="top" width="9.69%" headers="mcps1.2.4.1.2 "><p id="p13901052205318"><a name="p13901052205318"></a><a name="p13901052205318"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="72.54%" headers="mcps1.2.4.1.3 "><p id="p1290165213532"><a name="p1290165213532"></a><a name="p1290165213532"></a>BatchNorm接口能完成计算所需最小临时空间大小。为保证功能正确，接口计算时预留/申请的临时空间不能小于min的数值。最小空间为0表示计算不需要临时空间。</p>
</td>
</tr>
<tr id="row147105158182"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p0710171510188"><a name="p0710171510188"></a><a name="p0710171510188"></a>isBasicBlock</p>
</td>
<td class="cellrowborder" valign="top" width="9.69%" headers="mcps1.2.4.1.2 "><p id="p1871012153180"><a name="p1871012153180"></a><a name="p1871012153180"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.54%" headers="mcps1.2.4.1.3 "><p id="p5766412163619"><a name="p5766412163619"></a><a name="p5766412163619"></a>是否使能基本块，与BatchNorm接口一致。</p>
</td>
</tr>
</tbody>
</table>

**表2**  GetBatchNormNDTilingInfo接口参数列表：

<a name="table667463712434"></a>
<table><thead align="left"><tr id="row5674103712436"><th class="cellrowborder" valign="top" width="19.63196319631963%" id="mcps1.1.4.1.1"><p id="p16674193714437"><a name="p16674193714437"></a><a name="p16674193714437"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="10.76107610761076%" id="mcps1.1.4.1.2"><p id="p8674133764310"><a name="p8674133764310"></a><a name="p8674133764310"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="69.60696069606959%" id="mcps1.1.4.1.3"><p id="p1367413376436"><a name="p1367413376436"></a><a name="p1367413376436"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row1019612431301"><td class="cellrowborder" valign="top" width="19.63196319631963%" headers="mcps1.1.4.1.1 "><p id="p2858194711302"><a name="p2858194711302"></a><a name="p2858194711302"></a>srcShape</p>
</td>
<td class="cellrowborder" valign="top" width="10.76107610761076%" headers="mcps1.1.4.1.2 "><p id="p1619614303015"><a name="p1619614303015"></a><a name="p1619614303015"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="69.60696069606959%" headers="mcps1.1.4.1.3 "><p id="p93218541303"><a name="p93218541303"></a><a name="p93218541303"></a>输入数据inputX的shape信息[B, S, H]，S*H需要32B对齐。</p>
</td>
</tr>
<tr id="row15674237164319"><td class="cellrowborder" valign="top" width="19.63196319631963%" headers="mcps1.1.4.1.1 "><p id="p4711394483"><a name="p4711394483"></a><a name="p4711394483"></a>originSrcShape</p>
</td>
<td class="cellrowborder" valign="top" width="10.76107610761076%" headers="mcps1.1.4.1.2 "><p id="p267433711433"><a name="p267433711433"></a><a name="p267433711433"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="69.60696069606959%" headers="mcps1.1.4.1.3 "><p id="p7973722104817"><a name="p7973722104817"></a><a name="p7973722104817"></a>输入数据inputX的originshape信息[originB, originS, originH]。</p>
</td>
</tr>
<tr id="row19723118144412"><td class="cellrowborder" valign="top" width="19.63196319631963%" headers="mcps1.1.4.1.1 "><p id="p4723188154413"><a name="p4723188154413"></a><a name="p4723188154413"></a>stackBufferByteSize</p>
</td>
<td class="cellrowborder" valign="top" width="10.76107610761076%" headers="mcps1.1.4.1.2 "><p id="p15723148144416"><a name="p15723148144416"></a><a name="p15723148144416"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="69.60696069606959%" headers="mcps1.1.4.1.3 "><p id="p77236817446"><a name="p77236817446"></a><a name="p77236817446"></a>可供BatchNorm接口使用的空间大小，单位Byte。</p>
</td>
</tr>
<tr id="row44731452154416"><td class="cellrowborder" valign="top" width="19.63196319631963%" headers="mcps1.1.4.1.1 "><p id="p15473135218445"><a name="p15473135218445"></a><a name="p15473135218445"></a>typeSize</p>
</td>
<td class="cellrowborder" valign="top" width="10.76107610761076%" headers="mcps1.1.4.1.2 "><p id="p144731152144416"><a name="p144731152144416"></a><a name="p144731152144416"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="69.60696069606959%" headers="mcps1.1.4.1.3 "><p id="p647419528446"><a name="p647419528446"></a><a name="p647419528446"></a>输入数据类型的字节大小。</p>
</td>
</tr>
<tr id="row1017792113457"><td class="cellrowborder" valign="top" width="19.63196319631963%" headers="mcps1.1.4.1.1 "><p id="p8178321144519"><a name="p8178321144519"></a><a name="p8178321144519"></a>isReuseSource</p>
</td>
<td class="cellrowborder" valign="top" width="10.76107610761076%" headers="mcps1.1.4.1.2 "><p id="p16178152124518"><a name="p16178152124518"></a><a name="p16178152124518"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="69.60696069606959%" headers="mcps1.1.4.1.3 "><p id="p11389449156"><a name="p11389449156"></a><a name="p11389449156"></a>中间变量是否能够复用输入内存。该参数预留，传入默认值false即可。</p>
</td>
</tr>
<tr id="row8764163917458"><td class="cellrowborder" valign="top" width="19.63196319631963%" headers="mcps1.1.4.1.1 "><p id="p376410391458"><a name="p376410391458"></a><a name="p376410391458"></a>tilling</p>
</td>
<td class="cellrowborder" valign="top" width="10.76107610761076%" headers="mcps1.1.4.1.2 "><p id="p576417394455"><a name="p576417394455"></a><a name="p576417394455"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="69.60696069606959%" headers="mcps1.1.4.1.3 "><p id="p0764153917454"><a name="p0764153917454"></a><a name="p0764153917454"></a>输入数据的切分信息。</p>
</td>
</tr>
<tr id="row9105134681611"><td class="cellrowborder" valign="top" width="19.63196319631963%" headers="mcps1.1.4.1.1 "><p id="p1899220460164"><a name="p1899220460164"></a><a name="p1899220460164"></a>isBasicBlock</p>
</td>
<td class="cellrowborder" valign="top" width="10.76107610761076%" headers="mcps1.1.4.1.2 "><p id="p49920461166"><a name="p49920461166"></a><a name="p49920461166"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="69.60696069606959%" headers="mcps1.1.4.1.3 "><p id="p179921246191610"><a name="p179921246191610"></a><a name="p179921246191610"></a>是否使能基本块，与BatchNorm接口一致。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section2075135024716"></a>

-   GetBatchNormMaxMinTmpSize返回值为true/false，true表示成功拿到BatchNorm接口内部计算需要的最大和最小临时空间大小；false表示获取失败。
-   GetBatchNormNDTilingInfo返回类型为true/false，true表示成功拿到BatchNorm的Tiling各项参数值；false表示获取失败。

## 约束说明<a name="section92611953111217"></a>

无

## 调用示例<a name="section642mcpsimp"></a>

如下样例介绍了host侧获取Tiling参数的流程以及该参数如何在kernel侧使用。样例中输入Tensor的shape大小为\[16，16，16\]，输入的数据类型为half。

1.  将BatchNormTiling结构体参数增加至TilingData结构体，作为TilingData结构体的一个字段。

    ```
    BEGIN_TILING_DATA_DEF(TilingData)               // 注册一个tiling的类，以tiling的名字作为入参
      TILING_DATA_FIELD_DEF(uint32_t, tileNum);     // 添加tiling字段，每个核上总计算数据分块个数
      TILING_DATA_FIELD_DEF(uint32_t, bLength);     // 添加tiling字段，输入shape的b维度长度
      TILING_DATA_FIELD_DEF(uint32_t, sLength);     // 添加tiling字段，输入shape的s维度长度
      TILING_DATA_FIELD_DEF(uint32_t, hLength);     // 添加tiling字段，输入shape的h维度长度
      TILING_DATA_FIELD_DEF(uint32_t, originalBLength);     // 添加tiling字段，输入shape原始b维度长度
      ...                                           // 添加其他tiling字段
      TILING_DATA_FIELD_DEF_STRUCT(BatchNormTiling, batchNormTilingData); // 将BatchNormTiling结构体参数增加至TilingData结构体
    END_TILING_DATA_DEF;
    ```

2.  Tiling实现函数中，首先调用**GetBatchNormMaxMinTmpSize**接口获取BatchNorm接口能完成计算所需最大/最小临时空间大小，根据该范围结合实际的内存使用情况设置合适的空间大小，然后根据输入shape、剩余的可供计算的空间大小等信息获取BatchNorm kernel侧接口所需tiling参数。

    ```
    namespace optiling {
    const uint32_t NUM_BLOCKS = 8;
    const uint32_t TILE_NUM = 8;
    static ge::graphStatus TilingFunc(gert::TilingContext* context)
    {
        TilingData tiling;
        uint32_t totalLength = context->GetInputTensor(0)->GetShapeSize();
        context->SetBlockDim(NUM_BLOCKS);
        tiling.set_tileNum(TILE_NUM);
        // 设置其他Tiling参数
        ... 
        std::vector<int64_t> shapeVec = {16, 16, 16};//{b,s,h}
        std::vector<int64_t> originShapeVec = {15, 16, 16};//{originB,originS,originH}
        ge::Shape srcShape(shapeVec);
        ge::Shape originSrcShape(originShapeVec);
        uint32_t minSize = 0;
        uint32_t maxSize = 0;
        // 本样例中仅作为样例说明，通过GetBatchNormMaxMinTmpSize获取最小值并传入，来保证功能正确，开发者可以根据需要传入合适的空间大小
        AscendC::GetBatchNormMaxMinTmpSize(srcShape, originSrcShape, sizeof(half), false, maxSize, minSize, false);
        // 获取BatchNorm Tiling参数
        AscendC::GetBatchNormNDTilingInfo(srcShape, originSrcShape, minSize, sizeof(half), false, tiling.batchNormTilingData, false); 
         ... // 其他逻辑
        tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
        context->SetTilingKey(1);
        return ge::GRAPH_SUCCESS;
    }
    } // namespace optiling
    ```

3.  对应的kernel侧通过在核函数中调用GET\_TILING\_DATA获取TilingData，继而将TilingData中的BatchNormTiling信息传入BatchNorm接口参与计算。完整的kernel侧样例请参考[BatchNorm](BatchNorm.md)

    ```
    extern "C" __global__ __aicore__ void func_custom(GM_ADDR inputX_gm, GM_ADDR gamm_gm, GM_ADDR beta_gm, GM_ADDR output_gm, GM_ADDR outputMean_gm, GM_ADDR outputVariance_gm, GM_ADDR tiling)
    {
        GET_TILING_DATA(tilingData, tiling);
        KernelBatchnorm<half, false, false> op;
        op.Init(inputX_gm, gamm_gm, beta_gm, output_gm, outputMean_gm, outputVariance_gm, tilingData.batchNormTilingData);
        if (TILING_KEY_IS(1)) {
            op.Process();
        }
    }
    ```

