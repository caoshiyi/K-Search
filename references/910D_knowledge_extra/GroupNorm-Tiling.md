# GroupNorm Tiling<a name="ZH-CN_TOPIC_0000002554343885"></a>

## 功能说明<a name="section2092310566815"></a>

GroupNorm Tiling API用于获取GroupNorm kernel计算时所需的Tiling参数。获取Tiling参数主要分为如下两步：

1.  通过**GetGroupNormMaxMinTmpSize**获取GroupNorm接口计算所需最大和最小临时空间大小。

    kernel侧GroupNorm接口的计算需要开发者预留/申请临时空间，**GetGroupNormMaxMinTmpSize**用于在host侧获取预留/申请的最大最小临时空间大小，开发者基于此范围选择合适的空间大小作为Tiling参数传递到kernel侧使用。

    -   为保证功能正确，预留/申请的临时空间大小不能小于最小临时空间大小；
    -   在最小临时空间-最大临时空间范围内，随着临时空间增大，kernel侧接口计算性能会有一定程度的优化提升。为了达到更好的性能，开发者可以根据实际的内存使用情况进行空间预留/申请。

2.  通过**GetGroupNormNDTilingInfo**获取GroupNorm kernel侧接口所需tiling参数。

    GroupNorm Tiling结构体的定义如下，开发者无需关注该tiling结构的具体信息，只需要传递到kernel侧，传入GroupNorm高阶API接口，直接进行使用即可。

    ```
    struct GroupNormTiling {
        uint32_t n = 0;
        uint32_t c = 0;
        uint32_t hw = 0;
        uint32_t g = 0;
        uint32_t d = 0;
        uint32_t hwAlignSize = 0;
        uint32_t dhwAlignSize = 0;
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
        float factor = 0;
        bool smallShape = 0;
    };
    ```

## 函数原型<a name="section1517111135714"></a>

```
void GetGroupNormMaxMinTmpSize(const ge::Shape& srcShape, const uint32_t typeSize, const bool isReuseSource, const uint32_t groupNum, uint32_t& maxValue, uint32_t& minValue)
```

```
void GetGroupNormNDTilingInfo(const ge::Shape& srcShape, const uint32_t stackBufferSize, const uint32_t typeSize, const bool isReuseSource, const uint32_t groupNum, optiling::GroupNormTiling& tiling)
```

```
void GetGroupNormNDTilingInfo(const ge::Shape& srcShape, const uint32_t stackBufferSize, const uint32_t typeSize, const bool isReuseSource, const uint32_t groupNum, AscendC::tiling::GroupNormTiling& tiling)
```

## 参数说明<a name="section1557304615912"></a>

**表 1**  GetGroupNormMaxMinTmpSize接口参数列表

<a name="table729818506422"></a>
<table><thead align="left"><tr id="row11299950204217"><th class="cellrowborder" valign="top" width="17.98%" id="mcps1.2.4.1.1"><p id="p1029955044218"><a name="p1029955044218"></a><a name="p1029955044218"></a>接口</p>
</th>
<th class="cellrowborder" valign="top" width="12.8%" id="mcps1.2.4.1.2"><p id="p13216203215467"><a name="p13216203215467"></a><a name="p13216203215467"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="69.22%" id="mcps1.2.4.1.3"><p id="p1629911506421"><a name="p1629911506421"></a><a name="p1629911506421"></a>功能</p>
</th>
</tr>
</thead>
<tbody><tr id="row12299165018421"><td class="cellrowborder" valign="top" width="17.98%" headers="mcps1.2.4.1.1 "><p id="p1329915004219"><a name="p1329915004219"></a><a name="p1329915004219"></a>srcShape</p>
</td>
<td class="cellrowborder" valign="top" width="12.8%" headers="mcps1.2.4.1.2 "><p id="p521618329466"><a name="p521618329466"></a><a name="p521618329466"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="69.22%" headers="mcps1.2.4.1.3 "><p id="p199212536714"><a name="p199212536714"></a><a name="p199212536714"></a>输入数据inputX的shape信息[N, C, H, W]。</p>
</td>
</tr>
<tr id="row19299125011422"><td class="cellrowborder" valign="top" width="17.98%" headers="mcps1.2.4.1.1 "><p id="p1929916502421"><a name="p1929916502421"></a><a name="p1929916502421"></a>typeSize</p>
</td>
<td class="cellrowborder" valign="top" width="12.8%" headers="mcps1.2.4.1.2 "><p id="p172166321461"><a name="p172166321461"></a><a name="p172166321461"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="69.22%" headers="mcps1.2.4.1.3 "><p id="p8299150154218"><a name="p8299150154218"></a><a name="p8299150154218"></a>输入数据inputX的数据类型大小，单位为字节。比如输入的数据类型为half，此处应传入2。</p>
</td>
</tr>
<tr id="row5299125054217"><td class="cellrowborder" valign="top" width="17.98%" headers="mcps1.2.4.1.1 "><p id="p9777142884312"><a name="p9777142884312"></a><a name="p9777142884312"></a>isReuseSource</p>
</td>
<td class="cellrowborder" valign="top" width="12.8%" headers="mcps1.2.4.1.2 "><p id="p1221613214466"><a name="p1221613214466"></a><a name="p1221613214466"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="69.22%" headers="mcps1.2.4.1.3 "><p id="p277815284439"><a name="p277815284439"></a><a name="p277815284439"></a>中间变量是否能够复用输入内存。</p>
</td>
</tr>
<tr id="row3524438151015"><td class="cellrowborder" valign="top" width="17.98%" headers="mcps1.2.4.1.1 "><p id="p108231747151016"><a name="p108231747151016"></a><a name="p108231747151016"></a>groupNum</p>
</td>
<td class="cellrowborder" valign="top" width="12.8%" headers="mcps1.2.4.1.2 "><p id="p19525143812102"><a name="p19525143812102"></a><a name="p19525143812102"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="69.22%" headers="mcps1.2.4.1.3 "><p id="p5525173871019"><a name="p5525173871019"></a><a name="p5525173871019"></a>在C维度上的分组数。</p>
</td>
</tr>
<tr id="row6777152811436"><td class="cellrowborder" valign="top" width="17.98%" headers="mcps1.2.4.1.1 "><p id="p23791451102416"><a name="p23791451102416"></a><a name="p23791451102416"></a>maxValue</p>
</td>
<td class="cellrowborder" valign="top" width="12.8%" headers="mcps1.2.4.1.2 "><p id="p470071112510"><a name="p470071112510"></a><a name="p470071112510"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="69.22%" headers="mcps1.2.4.1.3 "><p id="p1288155462517"><a name="p1288155462517"></a><a name="p1288155462517"></a>输出GroupNorm接口所需的tiling信息（最大临时空间大小）。</p>
<p id="p1078374516308"><a name="p1078374516308"></a><a name="p1078374516308"></a>GroupNorm接口能完成计算所需的最大临时空间大小，超出该值的空间不会被该接口使用。在最小临时空间-最大临时空间范围内，随着临时空间增大，kernel侧接口计算性能会有一定程度的优化提升。为了达到更好的性能，开发者可以根据实际的内存使用情况进行空间预留/申请。</p>
<div class="note" id="note1275197121212"><a name="note1275197121212"></a><a name="note1275197121212"></a><span class="notetitle"> 说明： </span><div class="notebody"><p id="p1040915016493"><a name="p1040915016493"></a><a name="p1040915016493"></a>maxValue仅作为参考值，有可能大于<span id="ph1088254310583"><a name="ph1088254310583"></a><a name="ph1088254310583"></a>Unified Buffer</span>剩余空间的大小，该场景下，开发者需要根据<span id="ph20796650203718"><a name="ph20796650203718"></a><a name="ph20796650203718"></a>Unified Buffer</span>剩余空间的大小来选取合适的临时空间大小。</p>
</div></div>
</td>
</tr>
<tr id="row6563634154317"><td class="cellrowborder" valign="top" width="17.98%" headers="mcps1.2.4.1.1 "><p id="p956314345431"><a name="p956314345431"></a><a name="p956314345431"></a>minValue</p>
</td>
<td class="cellrowborder" valign="top" width="12.8%" headers="mcps1.2.4.1.2 "><p id="p42161432144610"><a name="p42161432144610"></a><a name="p42161432144610"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="69.22%" headers="mcps1.2.4.1.3 "><p id="p20807932162517"><a name="p20807932162517"></a><a name="p20807932162517"></a>输出GroupNorm接口所需的tiling信息（最小临时空间大小）。</p>
<p id="p281475823012"><a name="p281475823012"></a><a name="p281475823012"></a>GroupNorm接口能完成计算所需最小临时空间大小。为保证功能正确，接口计算时预留/申请的临时空间不能小于该数值。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  GetGroupNormNDTilingInfo接口参数列表：

<a name="table667463712434"></a>
<table><thead align="left"><tr id="row5674103712436"><th class="cellrowborder" valign="top" width="19.63196319631963%" id="mcps1.2.4.1.1"><p id="p16674193714437"><a name="p16674193714437"></a><a name="p16674193714437"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="13.451345134513451%" id="mcps1.2.4.1.2"><p id="p8674133764310"><a name="p8674133764310"></a><a name="p8674133764310"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="66.91669166916692%" id="mcps1.2.4.1.3"><p id="p1367413376436"><a name="p1367413376436"></a><a name="p1367413376436"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row15674237164319"><td class="cellrowborder" valign="top" width="19.63196319631963%" headers="mcps1.2.4.1.1 "><p id="p1467483715439"><a name="p1467483715439"></a><a name="p1467483715439"></a>srcShape</p>
</td>
<td class="cellrowborder" valign="top" width="13.451345134513451%" headers="mcps1.2.4.1.2 "><p id="p267433711433"><a name="p267433711433"></a><a name="p267433711433"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="66.91669166916692%" headers="mcps1.2.4.1.3 "><p id="p240633015261"><a name="p240633015261"></a><a name="p240633015261"></a>输入数据inputX的shape信息[N, C, H, W]。</p>
</td>
</tr>
<tr id="row19723118144412"><td class="cellrowborder" valign="top" width="19.63196319631963%" headers="mcps1.2.4.1.1 "><p id="p4723188154413"><a name="p4723188154413"></a><a name="p4723188154413"></a>stackBufferSize</p>
</td>
<td class="cellrowborder" valign="top" width="13.451345134513451%" headers="mcps1.2.4.1.2 "><p id="p15723148144416"><a name="p15723148144416"></a><a name="p15723148144416"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="66.91669166916692%" headers="mcps1.2.4.1.3 "><p id="p77236817446"><a name="p77236817446"></a><a name="p77236817446"></a>可供GroupNorm接口使用的空间大小，单位Byte。</p>
</td>
</tr>
<tr id="row44731452154416"><td class="cellrowborder" valign="top" width="19.63196319631963%" headers="mcps1.2.4.1.1 "><p id="p15473135218445"><a name="p15473135218445"></a><a name="p15473135218445"></a>typeSize</p>
</td>
<td class="cellrowborder" valign="top" width="13.451345134513451%" headers="mcps1.2.4.1.2 "><p id="p144731152144416"><a name="p144731152144416"></a><a name="p144731152144416"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="66.91669166916692%" headers="mcps1.2.4.1.3 "><p id="p597311610461"><a name="p597311610461"></a><a name="p597311610461"></a>输入的数据类型大小，单位为字节。比如输入的数据类型为half，此处应传入2。</p>
</td>
</tr>
<tr id="row1017792113457"><td class="cellrowborder" valign="top" width="19.63196319631963%" headers="mcps1.2.4.1.1 "><p id="p8178321144519"><a name="p8178321144519"></a><a name="p8178321144519"></a>isReuseSource</p>
</td>
<td class="cellrowborder" valign="top" width="13.451345134513451%" headers="mcps1.2.4.1.2 "><p id="p16178152124518"><a name="p16178152124518"></a><a name="p16178152124518"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="66.91669166916692%" headers="mcps1.2.4.1.3 "><p id="p18178172110459"><a name="p18178172110459"></a><a name="p18178172110459"></a>是否可以复用inputX的内存空间。</p>
</td>
</tr>
<tr id="row232016255297"><td class="cellrowborder" valign="top" width="19.63196319631963%" headers="mcps1.2.4.1.1 "><p id="p1632018252297"><a name="p1632018252297"></a><a name="p1632018252297"></a>groupNum</p>
</td>
<td class="cellrowborder" valign="top" width="13.451345134513451%" headers="mcps1.2.4.1.2 "><p id="p632082510296"><a name="p632082510296"></a><a name="p632082510296"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="66.91669166916692%" headers="mcps1.2.4.1.3 "><p id="p17320525132920"><a name="p17320525132920"></a><a name="p17320525132920"></a>在C维度上的分组数。</p>
</td>
</tr>
<tr id="row8764163917458"><td class="cellrowborder" valign="top" width="19.63196319631963%" headers="mcps1.2.4.1.1 "><p id="p376410391458"><a name="p376410391458"></a><a name="p376410391458"></a>tiling</p>
</td>
<td class="cellrowborder" valign="top" width="13.451345134513451%" headers="mcps1.2.4.1.2 "><p id="p576417394455"><a name="p576417394455"></a><a name="p576417394455"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="66.91669166916692%" headers="mcps1.2.4.1.3 "><p id="p0764153917454"><a name="p0764153917454"></a><a name="p0764153917454"></a>输入数据的切分信息。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section911633517310"></a>

无

## 约束说明<a name="section92611953111217"></a>

无

## 调用示例<a name="section3434182143212"></a>

如下样例介绍了host侧获取Tiling参数的流程以及该参数如何在kernel侧使用。样例中输入Tensor的shape大小为\[2，16，8, 8\]，输入的数据类型为half。

1.  将GroupNormTiling结构体参数增加至TilingData结构体，作为TilingData结构体的一个字段。

    ```
    BEGIN_TILING_DATA_DEF(TilingData)  // 注册一个tiling的类，以tiling的名字作为入参
      TILING_DATA_FIELD_DEF(uint32_t, n);  
      TILING_DATA_FIELD_DEF(uint32_t, c);     
      TILING_DATA_FIELD_DEF(uint32_t, h);     
      TILING_DATA_FIELD_DEF(uint32_t, w);     
      TILING_DATA_FIELD_DEF(uint32_t, group);
      // 添加其他tiling字段
      ...                                           
      TILING_DATA_FIELD_DEF_STRUCT(GroupNormTiling, GroupNormTilingData); // 将GroupNormTiling结构体参数增加至TilingData结构体
    END_TILING_DATA_DEF;
    ```

2.  Tiling实现函数中，首先调用**GetGroupNormMaxMinTmpSize**接口获取GroupNorm接口能完成计算所需最大/最小临时空间大小，根据该范围结合实际的内存使用情况设置合适的空间大小，然后根据输入shape、剩余的可供计算的空间大小等信息获取GroupNorm kernel侧接口所需tiling参数。

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
        std::vector<int64_t> shapeVec = {2, 16, 8, 8}; // {n, c, h, w}
        ge::Shape srcShape(shapeVec);
        uint32_t groupNum=4
        uint32_t minSize = 0;
        uint32_t maxSize = 0;
        // 本样例中仅作为样例说明，通过GetGroupNormMaxMinTmpSize接口获取GroupNorm接口能完成计算所需最大/最小临时空间大小，开发者可以根据该范围结合实际的内存使用情况设置合适的空间大小
        AscendC::GetGroupNormMaxMinTmpSize(srcShape, sizeof(half), false, groupNum, maxSize, minSize);
        // 获取GroupNorm Tiling参数
        AscendC::GetGroupNormNDTilingInfo(srcShape, maxSize, sizeof(half), false, groupNum, tiling.groupNormTilingData); 
         ... // 其他逻辑
        tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
        context->SetTilingKey(1);
        return ge::GRAPH_SUCCESS;
    }
    } // namespace optiling
    ```

3.  对应的kernel侧通过在核函数中调用GET\_TILING\_DATA获取TilingData，继而将TilingData中的GroupNorm Tiling信息传入GroupNorm接口参与计算。

    ```
    extern "C" __global__ __aicore__ void groupnorm_custom(GM_ADDR inputX_gm, GM_ADDR gamm_gm, GM_ADDR beta_gm, GM_ADDR output_gm, GM_ADDR outputMean_gm, GM_ADDR outputVariance_gm, GM_ADDR tiling)
    {
        GET_TILING_DATA(tilingData, tiling);
        KernelGroupNorm<half, false> op;
        op.Init(inputX_gm, gamm_gm, beta_gm, output_gm, outputMean_gm, outputVariance_gm, tilingData.groupNormTilingData);
        if (TILING_KEY_IS(1)) {
            op.Process();
        }
    }
    ```

