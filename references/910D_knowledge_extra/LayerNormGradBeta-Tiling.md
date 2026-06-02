# LayerNormGradBeta Tiling<a name="ZH-CN_TOPIC_0000002554424045"></a>

## 功能说明<a name="section618mcpsimp"></a>

LayerNormGradBeta Tiling的功能如下：

-   在host侧获取预留/申请的最大最小临时空间大小：

    kernel侧LayerNormGradBeta接口的计算需要开发者预留/申请临时空间，**GetLayerNormGradBetaMaxMinTmpSize**接口用于在host侧获取预留/申请的最大最小临时空间大小，开发者基于此范围选择合适的空间大小作为Tiling参数传递到kernel侧使用。

    -   为保证功能正确，预留/申请的临时空间大小不能小于最小临时空间大小；
    -   在最小临时空间-最大临时空间范围内，随着临时空间增大，kernel侧接口计算性能会有一定程度的优化提升。为了达到更好的性能，开发者可以根据实际的内存使用情况进行空间预留/申请。

-   通过**GetLayerNormGradBetaNDTilingInfo**获取LayerNormGradBeta kernel侧接口所需tiling参数，需要传入输入shape，剩余的可供LayerNormGradBeta接口计算的空间大小和计算的数据类型。

    LayerNormGradBeta Tiling结构体的定义如下，开发者无需关注该Tiling结构的具体信息，只需要传递到kernel侧，传入LayerNormGradBeta高阶API接口，直接进行使用即可。

    ```
    struct LayerNormGradBetaTiling {
        uint32_t stackBufferSize = 0;
        uint32_t bLength = 0;
        uint32_t sLength = 0;
        uint32_t hLength = 0;
        uint32_t originalHLength = 0;
        uint32_t bshLength = 0;
        uint32_t bsLength = 0;
        uint32_t oneCalSize = 0;
        uint32_t numberOfTmpBuf = 0;
        uint32_t loopRound = 0;
        uint32_t inputTailSize = 0;
        uint32_t inputTailPos = 0;
        uint32_t bsTailSize = 0;
        uint32_t bshCurLength = 0;
        uint32_t bsCurLength = 0;
        uint32_t gammaTempTensorPos = 0;
        uint32_t betaTempTensorPos = 0;
        uint32_t inputDyTmpTensorPos = 0;
        uint32_t resForGammaTmpTensorPos = 0;
        uint32_t reserved = 0;
    };
    ```

## 函数原型<a name="section620mcpsimp"></a>

```
void GetLayerNormGradBetaMaxMinTmpSize(const ge::Shape& srcShape, const uint32_t typeSize, const bool isReuseSource, uint32_t& maxValue, uint32_t& minValue)
```

```
void GetLayerNormGradBetaNDTilingInfo(const ge::Shape srcShape, const uint32_t stackBufferSize, const uint32_t typeSize, const bool isReuseSource, optiling::LayerNormGradBetaTiling& tiling)
```

```
void GetLayerNormGradBetaNDTilingInfo(const ge::Shape srcShape, const uint32_t stackBufferSize, const uint32_t typeSize, const bool isReuseSource, AscendC::tiling::LayerNormGradBetaTiling& tiling)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  GetLayerNormGradBetaMaxMinTmpSize接口参数列表

<a name="table199981212909"></a>
<table><thead align="left"><tr id="row10998212604"><th class="cellrowborder" valign="top" width="19.88888888888889%" id="mcps1.2.4.1.1"><p id="p399821219019"><a name="p399821219019"></a><a name="p399821219019"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="17.535353535353536%" id="mcps1.2.4.1.2"><p id="p1099811217011"><a name="p1099811217011"></a><a name="p1099811217011"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="62.575757575757585%" id="mcps1.2.4.1.3"><p id="p899841214017"><a name="p899841214017"></a><a name="p899841214017"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row299881216012"><td class="cellrowborder" valign="top" width="19.88888888888889%" headers="mcps1.2.4.1.1 "><p id="p818613581102"><a name="p818613581102"></a><a name="p818613581102"></a>srcShape</p>
</td>
<td class="cellrowborder" valign="top" width="17.535353535353536%" headers="mcps1.2.4.1.2 "><p id="p101867581400"><a name="p101867581400"></a><a name="p101867581400"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="62.575757575757585%" headers="mcps1.2.4.1.3 "><p id="p132353614110"><a name="p132353614110"></a><a name="p132353614110"></a>输入数据inputDy的shape信息{B, S, storageHLength, originHLength}，包括当前输入的inputDy的shape信息，以及地址对齐前（如存在H轴补齐操作）的原有shape信息 。</p>
<p id="p7400192219571"><a name="p7400192219571"></a><a name="p7400192219571"></a>在API支持的场景下，storageHLength和originHLength保持一致。</p>
</td>
</tr>
<tr id="row499814129020"><td class="cellrowborder" valign="top" width="19.88888888888889%" headers="mcps1.2.4.1.1 "><p id="p121861158509"><a name="p121861158509"></a><a name="p121861158509"></a>typeSize</p>
</td>
<td class="cellrowborder" valign="top" width="17.535353535353536%" headers="mcps1.2.4.1.2 "><p id="p918665815018"><a name="p918665815018"></a><a name="p918665815018"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="62.575757575757585%" headers="mcps1.2.4.1.3 "><p id="p597311610461"><a name="p597311610461"></a><a name="p597311610461"></a>输入的数据类型大小，单位为字节。比如输入的数据类型为half，此处应传入2。</p>
</td>
</tr>
<tr id="row41973587135"><td class="cellrowborder" valign="top" width="19.88888888888889%" headers="mcps1.2.4.1.1 "><p id="p9777142884312"><a name="p9777142884312"></a><a name="p9777142884312"></a>isReuseSource</p>
</td>
<td class="cellrowborder" valign="top" width="17.535353535353536%" headers="mcps1.2.4.1.2 "><p id="p1221613214466"><a name="p1221613214466"></a><a name="p1221613214466"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="62.575757575757585%" headers="mcps1.2.4.1.3 "><p id="p277815284439"><a name="p277815284439"></a><a name="p277815284439"></a>是否复用源操作数的内存空间，与<a href="LayerNorm.md">LayerNorm</a>接口一致。</p>
</td>
</tr>
<tr id="row6998112905"><td class="cellrowborder" valign="top" width="19.88888888888889%" headers="mcps1.2.4.1.1 "><p id="p141874585012"><a name="p141874585012"></a><a name="p141874585012"></a>maxValue</p>
</td>
<td class="cellrowborder" valign="top" width="17.535353535353536%" headers="mcps1.2.4.1.2 "><p id="p718713586016"><a name="p718713586016"></a><a name="p718713586016"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="62.575757575757585%" headers="mcps1.2.4.1.3 "><p id="p22125283118"><a name="p22125283118"></a><a name="p22125283118"></a>LayerNormGradBeta接口能完成计算所需的最大临时空间大小，超出该值的空间不会被该接口使用。在最小临时空间-最大临时空间范围内，随着临时空间增大，kernel侧接口计算性能会有一定程度的优化提升。为了达到更好的性能，开发者可以根据实际的内存使用情况进行空间预留/申请。最大空间大小为0表示计算不需要临时空间。</p>
<div class="note" id="note1275197121212"><a name="note1275197121212"></a><a name="note1275197121212"></a><span class="notetitle"> 说明： </span><div class="notebody"><p id="p1040915016493"><a name="p1040915016493"></a><a name="p1040915016493"></a>maxValue仅作为参考值，有可能大于<span id="ph1088254310583"><a name="ph1088254310583"></a><a name="ph1088254310583"></a>Unified Buffer</span>剩余空间的大小，该场景下，开发者需要根据<span id="ph20796650203718"><a name="ph20796650203718"></a><a name="ph20796650203718"></a>Unified Buffer</span>剩余空间的大小来选取合适的临时空间大小。</p>
</div></div>
</td>
</tr>
<tr id="row169981512609"><td class="cellrowborder" valign="top" width="19.88888888888889%" headers="mcps1.2.4.1.1 "><p id="p18187175818015"><a name="p18187175818015"></a><a name="p18187175818015"></a>minValue</p>
</td>
<td class="cellrowborder" valign="top" width="17.535353535353536%" headers="mcps1.2.4.1.2 "><p id="p41876587019"><a name="p41876587019"></a><a name="p41876587019"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="62.575757575757585%" headers="mcps1.2.4.1.3 "><p id="p0235146313"><a name="p0235146313"></a><a name="p0235146313"></a>LayerNormGradBeta接口能完成计算所需最小临时空间大小。为保证功能正确，接口计算时预留/申请的临时空间不能小于该数值。最小空间大小为0表示计算不需要临时空间。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  GetLayerNormGradBetaNDTilingInfo接口参数列表

<a name="table112119352311"></a>
<table><thead align="left"><tr id="row37233533112"><th class="cellrowborder" valign="top" width="19.88888888888889%" id="mcps1.2.4.1.1"><p id="p2723352314"><a name="p2723352314"></a><a name="p2723352314"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="17.535353535353536%" id="mcps1.2.4.1.2"><p id="p77233510315"><a name="p77233510315"></a><a name="p77233510315"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="62.575757575757585%" id="mcps1.2.4.1.3"><p id="p1972163511311"><a name="p1972163511311"></a><a name="p1972163511311"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row3721435203117"><td class="cellrowborder" valign="top" width="19.88888888888889%" headers="mcps1.2.4.1.1 "><p id="p57253573118"><a name="p57253573118"></a><a name="p57253573118"></a>srcShape</p>
</td>
<td class="cellrowborder" valign="top" width="17.535353535353536%" headers="mcps1.2.4.1.2 "><p id="p1872435143118"><a name="p1872435143118"></a><a name="p1872435143118"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="62.575757575757585%" headers="mcps1.2.4.1.3 "><p id="p12721335173113"><a name="p12721335173113"></a><a name="p12721335173113"></a>输入数据inputDy的shape信息，包括当前输入的shape信息，以及地址对齐前的原有shape信息。</p>
</td>
</tr>
<tr id="row572173593115"><td class="cellrowborder" valign="top" width="19.88888888888889%" headers="mcps1.2.4.1.1 "><p id="p1672173513311"><a name="p1672173513311"></a><a name="p1672173513311"></a>stackBufferSize</p>
</td>
<td class="cellrowborder" valign="top" width="17.535353535353536%" headers="mcps1.2.4.1.2 "><p id="p37218352311"><a name="p37218352311"></a><a name="p37218352311"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="62.575757575757585%" headers="mcps1.2.4.1.3 "><p id="p173163515318"><a name="p173163515318"></a><a name="p173163515318"></a>可供接口使用的空间大小，单位为元素个数。</p>
</td>
</tr>
<tr id="row17353573113"><td class="cellrowborder" valign="top" width="19.88888888888889%" headers="mcps1.2.4.1.1 "><p id="p9731035153110"><a name="p9731035153110"></a><a name="p9731035153110"></a>typeSize</p>
</td>
<td class="cellrowborder" valign="top" width="17.535353535353536%" headers="mcps1.2.4.1.2 "><p id="p1373123511317"><a name="p1373123511317"></a><a name="p1373123511317"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="62.575757575757585%" headers="mcps1.2.4.1.3 "><p id="p44243691017"><a name="p44243691017"></a><a name="p44243691017"></a>输入的数据类型大小，单位为字节。比如输入的数据类型为half，此处应传入2。</p>
</td>
</tr>
<tr id="row173193543118"><td class="cellrowborder" valign="top" width="19.88888888888889%" headers="mcps1.2.4.1.1 "><p id="p1073163583112"><a name="p1073163583112"></a><a name="p1073163583112"></a>isReuseSource</p>
</td>
<td class="cellrowborder" valign="top" width="17.535353535353536%" headers="mcps1.2.4.1.2 "><p id="p1673143573114"><a name="p1673143573114"></a><a name="p1673143573114"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="62.575757575757585%" headers="mcps1.2.4.1.3 "><p id="p17313519318"><a name="p17313519318"></a><a name="p17313519318"></a>是否可以复用inputDy的内存空间。</p>
</td>
</tr>
<tr id="row147393583112"><td class="cellrowborder" valign="top" width="19.88888888888889%" headers="mcps1.2.4.1.1 "><p id="p9734352314"><a name="p9734352314"></a><a name="p9734352314"></a>tiling</p>
</td>
<td class="cellrowborder" valign="top" width="17.535353535353536%" headers="mcps1.2.4.1.2 "><p id="p773135133113"><a name="p773135133113"></a><a name="p773135133113"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="62.575757575757585%" headers="mcps1.2.4.1.3 "><p id="p373103563110"><a name="p373103563110"></a><a name="p373103563110"></a>输入数据的切分信息。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section92611953111217"></a>

无

## 调用示例<a name="section94691236101419"></a>

如下样例介绍了使用LayerNormGradBeta高阶API时host侧获取Tiling参数的流程以及该参数如何在kernel侧使用。样例中输入Tensor的shape大小为\[2, 16, 64\]，输入的数据类型为half。

1.  将LayerNormGradBetaTiling结构体参数增加至TilingData结构体，作为TilingData结构体的一个字段。

    ```
    BEGIN_TILING_DATA_DEF(TilingData)               // 注册一个tiling的类，以tiling的名字作为入参
      TILING_DATA_FIELD_DEF(uint32_t, totalLength); // 添加tiling字段，总计算数据量
      TILING_DATA_FIELD_DEF(uint32_t, tileNum);     // 添加tiling字段，每个核上总计算数据分块个数
      ...                                           // 添加其他tiling字段
      TILING_DATA_FIELD_DEF_STRUCT(LayerNormGradBetaTiling, layernormGradBetaTilingData); // 将LayerNormGradBetaTiling结构体参数增加至TilingData结构体
    END_TILING_DATA_DEF;
    ```

2.  Tiling实现函数中，首先调用GetLayerNormGradBetaMaxMinTmpSize接口获取LayerNormGradBeta接口能完成计算所需最大/最小临时空间大小，根据该范围结合实际的内存使用情况设置合适的空间大小，然后调用GetLayerNormGradBetaNDTilingInfo接口根据输入shape、剩余的可供计算的空间大小等信息获取LayerNormGradBeta kernel侧接口所需tiling参数。

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
        // 本样例中仅作为样例说明，通过GetLayerNormGradBetaMaxMinTmpSize获取最小值并传入，来保证功能正确，开发者可以根据需要传入合适的空间大小
        uint32_t max;
        uint32_t min;
        AscendC::GetLayerNormGradBetaMaxMinTmpSize(srcShape, sizeof(half), false, max, min);
        // 获取LayerNormGradBeta Tiling参数
        AscendC::GetLayerNormGradBetaNDTilingInfo(srcShape, min, sizeof(half), false, tiling.layernormGradBetaTilingData); 
         ... // 其他逻辑
        tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
        context->SetTilingKey(1);
        return ge::GRAPH_SUCCESS;
    }
    } // namespace optiling
    ```

3.  对应的kernel侧通过在核函数中调用GET\_TILING\_DATA获取TilingData，继而将TilingData中的LayerNormGradBetaTiling信息传入LayerNormGradBeta接口参与计算。完整的kernel侧样例请参考[LayerNormGradBeta](LayerNormGradBeta.md)。

    ```
    extern "C" __global__ __aicore__ void func_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
    {
        GET_TILING_DATA(tilingData, tiling);
        KernelFunc op;
        op.Init(x, y, z, tilingData.totalLength, tilingData.tileNum,tilingData.layernormGradBetaTilingData);
        if (TILING_KEY_IS(1)) {
            op.Process();
        }
    }
    ```

