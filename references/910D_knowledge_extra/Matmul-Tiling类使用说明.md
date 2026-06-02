# Matmul Tiling类使用说明<a name="ZH-CN_TOPIC_0000002523304454"></a>

Ascend C提供一组Matmul  [Tiling](基础知识.md#zh-cn_topic_0000001622194138_section68451031218)  API，方便用户获取[Matmul kernel计算](Matmul使用说明.md#li5878185413338)时所需的Tiling参数。用户只需要传入A/B/C矩阵的Position位置、Format格式和DType数据类型等信息，调用API接口，即可获取到[Init](Init-106.md)中TCubeTiling结构体中的相关参数。

Matmul Tiling API分为Matmul单核Tiling接口、多核Tiling接口和BatchMatmul Tiling接口，分别用于Matmul单核计算、多核计算和BatchMatmul计算场景。获取Tiling参数的流程如下：

1.  创建一个单核Tiling对象，或多核Tiling对象，或BatchMatmul Tiling对象。
2.  设置A、B、C、Bias的参数类型信息；M、N、Ka、Kb形状信息等。
3.  调用[GetTiling](GetTiling.md)接口，获取Tiling信息。

使用Matmul单核Tiling接口、多核Tiling接口和BatchMatmul Tiling接口获取Tiling参数的样例如下：

-   Matmul单核Tiling

    ```
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    matmul_tiling::MatmulApiTiling tiling(ascendcPlatform); 
    // 设置A、B、C、Bias矩阵Position、Format、DType信息
    tiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
    tiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);   
    tiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);   
    tiling.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);   
    tiling.SetShape(1024, 1024, 1024); // 设置单核计算的M、N、K大小
    tiling.SetOrgShape(1024, 1024, 1024); // 设置原始输入M、N、K大小，单核Tiling与SetShape一致。若Ka,Kb不等长时，设置tiling.SetOrgShape(1024, 1024, 1024, 1280)   
    tiling.EnableBias(true); // 设置matmul计算包含bias
    tiling.SetBufferSpace(-1, -1, -1);  // 设定允许使用的空间，缺省使用该AI处理器所有空间
    optiling::TCubeTiling tilingData;   
    int64_t ret = tiling.GetTiling(tilingData);    // if ret = -1, get tiling failed
    ```

-   Matmul多核Tiling

    ```
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    matmul_tiling::MultiCoreMatmulTiling tiling(ascendcPlatform); 
    tiling.SetDim(1); // 设置参与计算的核数为1
    tiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);   
    tiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);   
    tiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);   
    tiling.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);   
    tiling.SetShape(1024, 1024, 1024);   
    tiling.SetSingleShape(1024, 1024, 1024);
    tiling.SetOrgShape(1024, 1024, 1024); 
    tiling.EnableBias(true);   
    tiling.SetBufferSpace(-1, -1, -1);  // 设定允许使用的空间，缺省使用该AI处理器所有空间
    optiling::TCubeTiling tilingData;   
    int64_t ret = tiling.GetTiling(tilingData);    // if ret = -1, get tiling failed 
    ```

-   BatchMatmul Tiling

    ```
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    matmul_tiling::BatchMatmulTiling bmmTiling(ascendcPlatform); 
      
    bmmTiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);   
    bmmTiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);   
    bmmTiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);   
    bmmTiling.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);   
    bmmTiling.EnableBias(true);
    bmmTiling.SetShape(64, 48, 32);   
    bmmTiling.SetSingleShape(64, 48, 32);
    bmmTiling.SetOrgShape(64, 48, 32); 
    // Layout类型为NORMAL时,通过SetBatchInfoForNormal设置A、B、C矩阵的Layout轴信息
    bmmTiling.SetBatchInfoForNormal(2, 2, 64, 48, 32);
    // Layout类型为BSNGD、SBNGD、BNGS1S2时, 通过SetALayout、SetBLayout、SetCLayout设置A、B、C矩阵的Layout轴信息
    // bmmTiling.SetALayout(3, 64, 2, 2, 32);
    // bmmTiling.SetBLayout(3, 32, 2, 2, 48);
    // bmmTiling.SetCLayout(3, 64, 2, 2, 48);
    bmmTiling.SetBatchNum(2);
    bmmTiling.SetBufferSpace(-1, -1, -1);  // 设定允许使用的空间，缺省使用该AI处理器所有空间
    optiling::TCubeTiling tilingData;
    int64_t ret = bmmTiling.GetTiling(tilingData);    // if ret = -1, get tiling failed
    ```

接口列表如下：

**表 1**  MatmulApiTiling/MultiCoreMatmulTiling/BatchMatmulTiling共有接口列表

<a name="table171406364408"></a>
<table><thead align="left"><tr id="row21406365408"><th class="cellrowborder" valign="top" width="30.75%" id="mcps1.2.3.1.1"><p id="p17140836184013"><a name="p17140836184013"></a><a name="p17140836184013"></a>接口</p>
</th>
<th class="cellrowborder" valign="top" width="69.25%" id="mcps1.2.3.1.2"><p id="p141405369409"><a name="p141405369409"></a><a name="p141405369409"></a>功能</p>
</th>
</tr>
</thead>
<tbody><tr id="row31417367406"><td class="cellrowborder" valign="top" width="30.75%" headers="mcps1.2.3.1.1 "><p id="p12604203117452"><a name="p12604203117452"></a><a name="p12604203117452"></a>SetAType</p>
</td>
<td class="cellrowborder" valign="top" width="69.25%" headers="mcps1.2.3.1.2 "><p id="p14604183114514"><a name="p14604183114514"></a><a name="p14604183114514"></a>设置A矩阵的位置，数据格式，数据类型，是否转置等信息。</p>
</td>
</tr>
<tr id="row8141936124012"><td class="cellrowborder" valign="top" width="30.75%" headers="mcps1.2.3.1.1 "><p id="p136048312455"><a name="p136048312455"></a><a name="p136048312455"></a>SetBType</p>
</td>
<td class="cellrowborder" valign="top" width="69.25%" headers="mcps1.2.3.1.2 "><p id="p1960493174514"><a name="p1960493174514"></a><a name="p1960493174514"></a>设置B矩阵的位置，数据格式，数据类型，是否转置等信息。</p>
</td>
</tr>
<tr id="row14141536124015"><td class="cellrowborder" valign="top" width="30.75%" headers="mcps1.2.3.1.1 "><p id="p1604153117452"><a name="p1604153117452"></a><a name="p1604153117452"></a>SetCType</p>
</td>
<td class="cellrowborder" valign="top" width="69.25%" headers="mcps1.2.3.1.2 "><p id="p146041131184511"><a name="p146041131184511"></a><a name="p146041131184511"></a>设置C矩阵的位置，数据格式，数据类型等信息。</p>
</td>
</tr>
<tr id="row778611272492"><td class="cellrowborder" valign="top" width="30.75%" headers="mcps1.2.3.1.1 "><p id="p27861927164912"><a name="p27861927164912"></a><a name="p27861927164912"></a>SetDequantType</p>
</td>
<td class="cellrowborder" valign="top" width="69.25%" headers="mcps1.2.3.1.2 "><p id="p1778642714492"><a name="p1778642714492"></a><a name="p1778642714492"></a>设置反量化的模式。</p>
</td>
</tr>
<tr id="row314119360403"><td class="cellrowborder" valign="top" width="30.75%" headers="mcps1.2.3.1.1 "><p id="p260433117452"><a name="p260433117452"></a><a name="p260433117452"></a>SetBiasType</p>
</td>
<td class="cellrowborder" valign="top" width="69.25%" headers="mcps1.2.3.1.2 "><p id="p196041331194515"><a name="p196041331194515"></a><a name="p196041331194515"></a>设置Bias的位置，数据格式，数据类型等信息。</p>
</td>
</tr>
<tr id="row1203167173711"><td class="cellrowborder" valign="top" width="30.75%" headers="mcps1.2.3.1.1 "><p id="p620357203716"><a name="p620357203716"></a><a name="p620357203716"></a>SetShape</p>
</td>
<td class="cellrowborder" valign="top" width="69.25%" headers="mcps1.2.3.1.2 "><p id="p92032763717"><a name="p92032763717"></a><a name="p92032763717"></a>设置Matmul单次计算的形状singleM、singleN、singleK，单位为元素个数。</p>
</td>
</tr>
<tr id="row71417369409"><td class="cellrowborder" valign="top" width="30.75%" headers="mcps1.2.3.1.1 "><p id="p18605143154510"><a name="p18605143154510"></a><a name="p18605143154510"></a>SetOrgShape</p>
</td>
<td class="cellrowborder" valign="top" width="69.25%" headers="mcps1.2.3.1.2 "><p id="p8605631114511"><a name="p8605631114511"></a><a name="p8605631114511"></a>设置Matmul计算时的原始完整的形状M、N、Ka、Kb，单位为元素个数。</p>
</td>
</tr>
<tr id="row782444416499"><td class="cellrowborder" valign="top" width="30.75%" headers="mcps1.2.3.1.1 "><p id="p582517448496"><a name="p582517448496"></a><a name="p582517448496"></a>SetALayout</p>
</td>
<td class="cellrowborder" valign="top" width="69.25%" headers="mcps1.2.3.1.2 "><p id="p482574444915"><a name="p482574444915"></a><a name="p482574444915"></a>设置A矩阵的Layout轴信息。</p>
</td>
</tr>
<tr id="row15874135284912"><td class="cellrowborder" valign="top" width="30.75%" headers="mcps1.2.3.1.1 "><p id="p1387475218491"><a name="p1387475218491"></a><a name="p1387475218491"></a>SetBLayout</p>
</td>
<td class="cellrowborder" valign="top" width="69.25%" headers="mcps1.2.3.1.2 "><p id="p1387410524496"><a name="p1387410524496"></a><a name="p1387410524496"></a>设置B矩阵的Layout轴信息。</p>
</td>
</tr>
<tr id="row1232275810492"><td class="cellrowborder" valign="top" width="30.75%" headers="mcps1.2.3.1.1 "><p id="p33231958154918"><a name="p33231958154918"></a><a name="p33231958154918"></a>SetCLayout</p>
</td>
<td class="cellrowborder" valign="top" width="69.25%" headers="mcps1.2.3.1.2 "><p id="p6323458154911"><a name="p6323458154911"></a><a name="p6323458154911"></a>设置C矩阵的Layout轴信息。</p>
</td>
</tr>
<tr id="row122061148506"><td class="cellrowborder" valign="top" width="30.75%" headers="mcps1.2.3.1.1 "><p id="p1120619410502"><a name="p1120619410502"></a><a name="p1120619410502"></a>SetBatchInfoForNormal</p>
</td>
<td class="cellrowborder" valign="top" width="69.25%" headers="mcps1.2.3.1.2 "><p id="p1620615455016"><a name="p1620615455016"></a><a name="p1620615455016"></a>设置A/B矩阵的M/N/K轴信息，以及A/B矩阵各自的Batch数。</p>
</td>
</tr>
<tr id="row01876245504"><td class="cellrowborder" valign="top" width="30.75%" headers="mcps1.2.3.1.1 "><p id="p1188142416506"><a name="p1188142416506"></a><a name="p1188142416506"></a>SetBatchNum</p>
</td>
<td class="cellrowborder" valign="top" width="69.25%" headers="mcps1.2.3.1.2 "><p id="p418862415015"><a name="p418862415015"></a><a name="p418862415015"></a>设置多Batch计算的最大Batch数。</p>
</td>
</tr>
<tr id="row73731406507"><td class="cellrowborder" valign="top" width="30.75%" headers="mcps1.2.3.1.1 "><p id="p1137374018504"><a name="p1137374018504"></a><a name="p1137374018504"></a>EnableBias</p>
</td>
<td class="cellrowborder" valign="top" width="69.25%" headers="mcps1.2.3.1.2 "><p id="p12374204013505"><a name="p12374204013505"></a><a name="p12374204013505"></a>设置Bias是否参与运算。</p>
</td>
</tr>
<tr id="row16141936194019"><td class="cellrowborder" valign="top" width="30.75%" headers="mcps1.2.3.1.1 "><p id="p760510318455"><a name="p760510318455"></a><a name="p760510318455"></a>SetBias</p>
</td>
<td class="cellrowborder" valign="top" width="69.25%" headers="mcps1.2.3.1.2 "><p id="p2060512314452"><a name="p2060512314452"></a><a name="p2060512314452"></a>设置Bias是否参与运算。建议使用EnableBias接口。</p>
</td>
</tr>
<tr id="row428123912408"><td class="cellrowborder" valign="top" width="30.75%" headers="mcps1.2.3.1.1 "><p id="p3605183114513"><a name="p3605183114513"></a><a name="p3605183114513"></a>SetFixSplit</p>
</td>
<td class="cellrowborder" valign="top" width="69.25%" headers="mcps1.2.3.1.2 "><p id="p36059313452"><a name="p36059313452"></a><a name="p36059313452"></a>设置固定的baseM、baseN、baseK，单位为元素个数。</p>
</td>
</tr>
<tr id="row3260124104015"><td class="cellrowborder" valign="top" width="30.75%" headers="mcps1.2.3.1.1 "><p id="p9605143194511"><a name="p9605143194511"></a><a name="p9605143194511"></a>SetBufferSpace</p>
</td>
<td class="cellrowborder" valign="top" width="69.25%" headers="mcps1.2.3.1.2 "><p id="p16051311452"><a name="p16051311452"></a><a name="p16051311452"></a>设置Matmul计算时可用的L1/L0C/UB空间大小，单位为字节。</p>
</td>
</tr>
<tr id="row421712438402"><td class="cellrowborder" valign="top" width="30.75%" headers="mcps1.2.3.1.1 "><p id="p20605133194520"><a name="p20605133194520"></a><a name="p20605133194520"></a>SetTraverse</p>
</td>
<td class="cellrowborder" valign="top" width="69.25%" headers="mcps1.2.3.1.2 "><p id="p126052314457"><a name="p126052314457"></a><a name="p126052314457"></a>设置遍历方式，M轴优先还是N轴优先。</p>
</td>
</tr>
<tr id="row285815120401"><td class="cellrowborder" valign="top" width="30.75%" headers="mcps1.2.3.1.1 "><p id="p176051831144520"><a name="p176051831144520"></a><a name="p176051831144520"></a>SetMadType</p>
</td>
<td class="cellrowborder" valign="top" width="69.25%" headers="mcps1.2.3.1.2 "><p id="p186053312458"><a name="p186053312458"></a><a name="p186053312458"></a>设置是否使能HF32模式。<strong id="b161112357918"><a name="b161112357918"></a><a name="b161112357918"></a>当前版本暂不支持。</strong></p>
</td>
</tr>
<tr id="row59632538401"><td class="cellrowborder" valign="top" width="30.75%" headers="mcps1.2.3.1.1 "><p id="p76061431144511"><a name="p76061431144511"></a><a name="p76061431144511"></a>SetSplitRange</p>
</td>
<td class="cellrowborder" valign="top" width="69.25%" headers="mcps1.2.3.1.2 "><p id="p1160633174520"><a name="p1160633174520"></a><a name="p1160633174520"></a>设置baseM/baseN/baseK的最大值和最小值。</p>
</td>
</tr>
<tr id="row11171174518"><td class="cellrowborder" valign="top" width="30.75%" headers="mcps1.2.3.1.1 "><p id="p1817378510"><a name="p1817378510"></a><a name="p1817378510"></a>SetMatmulConfigParams</p>
</td>
<td class="cellrowborder" valign="top" width="69.25%" headers="mcps1.2.3.1.2 "><p id="p1917117125113"><a name="p1917117125113"></a><a name="p1917117125113"></a>自定义设置MatmulConfig参数。</p>
</td>
</tr>
<tr id="row11980195524010"><td class="cellrowborder" valign="top" width="30.75%" headers="mcps1.2.3.1.1 "><p id="p3606113144512"><a name="p3606113144512"></a><a name="p3606113144512"></a>SetDoubleBuffer</p>
</td>
<td class="cellrowborder" valign="top" width="69.25%" headers="mcps1.2.3.1.2 "><p id="p3606231164512"><a name="p3606231164512"></a><a name="p3606231164512"></a>设置A/B/C/Bias是否使能double buffer功能，以及是否需要做ND2NZ或者NZ2ND的转换。<strong id="b1948724017223"><a name="b1948724017223"></a><a name="b1948724017223"></a>该接口为预留接口，当前版本暂不支持。</strong></p>
</td>
</tr>
<tr id="row1577115235456"><td class="cellrowborder" valign="top" width="30.75%" headers="mcps1.2.3.1.1 "><p id="p15606931194513"><a name="p15606931194513"></a><a name="p15606931194513"></a>GetBaseM</p>
</td>
<td class="cellrowborder" valign="top" width="69.25%" headers="mcps1.2.3.1.2 "><p id="p2606163174514"><a name="p2606163174514"></a><a name="p2606163174514"></a>获取baseM值。</p>
</td>
</tr>
<tr id="row171211624114511"><td class="cellrowborder" valign="top" width="30.75%" headers="mcps1.2.3.1.1 "><p id="p36061631154518"><a name="p36061631154518"></a><a name="p36061631154518"></a>GetBaseN</p>
</td>
<td class="cellrowborder" valign="top" width="69.25%" headers="mcps1.2.3.1.2 "><p id="p1760663115451"><a name="p1760663115451"></a><a name="p1760663115451"></a>获取baseN值。</p>
</td>
</tr>
<tr id="row11266224194515"><td class="cellrowborder" valign="top" width="30.75%" headers="mcps1.2.3.1.1 "><p id="p166061231114516"><a name="p166061231114516"></a><a name="p166061231114516"></a>GetBaseK</p>
</td>
<td class="cellrowborder" valign="top" width="69.25%" headers="mcps1.2.3.1.2 "><p id="p1960693114458"><a name="p1960693114458"></a><a name="p1960693114458"></a>获取baseK值。</p>
</td>
</tr>
<tr id="row9606162416458"><td class="cellrowborder" valign="top" width="30.75%" headers="mcps1.2.3.1.1 "><p id="p86077315455"><a name="p86077315455"></a><a name="p86077315455"></a>GetTiling</p>
</td>
<td class="cellrowborder" valign="top" width="69.25%" headers="mcps1.2.3.1.2 "><p id="p118051458544"><a name="p118051458544"></a><a name="p118051458544"></a>获取Tiling参数。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  MultiCoreMatmulTiling其他接口

<a name="table183308213484"></a>
<table><thead align="left"><tr id="row53302021487"><th class="cellrowborder" valign="top" width="18.17%" id="mcps1.2.3.1.1"><p id="p5330625481"><a name="p5330625481"></a><a name="p5330625481"></a>接口</p>
</th>
<th class="cellrowborder" valign="top" width="81.83%" id="mcps1.2.3.1.2"><p id="p23301123487"><a name="p23301123487"></a><a name="p23301123487"></a>功能</p>
</th>
</tr>
</thead>
<tbody><tr id="row83305244818"><td class="cellrowborder" valign="top" width="18.17%" headers="mcps1.2.3.1.1 "><p id="p187403136489"><a name="p187403136489"></a><a name="p187403136489"></a>SetDim</p>
</td>
<td class="cellrowborder" valign="top" width="81.83%" headers="mcps1.2.3.1.2 "><p id="p9740141311481"><a name="p9740141311481"></a><a name="p9740141311481"></a>设置多核Matmul时，可以参与运算的核数。</p>
</td>
</tr>
<tr id="row133314204813"><td class="cellrowborder" valign="top" width="18.17%" headers="mcps1.2.3.1.1 "><p id="p774061364813"><a name="p774061364813"></a><a name="p774061364813"></a>SetSingleRange</p>
</td>
<td class="cellrowborder" valign="top" width="81.83%" headers="mcps1.2.3.1.2 "><p id="p19740213144813"><a name="p19740213144813"></a><a name="p19740213144813"></a>设置singleCoreM/singleCoreN/singleCoreK的最大值与最小值，单位为元素个数。</p>
</td>
</tr>
<tr id="row13589349481"><td class="cellrowborder" valign="top" width="18.17%" headers="mcps1.2.3.1.1 "><p id="p1292210204810"><a name="p1292210204810"></a><a name="p1292210204810"></a>SetSingleShape</p>
</td>
<td class="cellrowborder" valign="top" width="81.83%" headers="mcps1.2.3.1.2 "><p id="p39321014488"><a name="p39321014488"></a><a name="p39321014488"></a>设置Matmul单核计算的形状singleCoreM、singleCoreN、singleCoreK，单位为元素个数。</p>
</td>
</tr>
<tr id="row53312213488"><td class="cellrowborder" valign="top" width="18.17%" headers="mcps1.2.3.1.1 "><p id="p1741131384818"><a name="p1741131384818"></a><a name="p1741131384818"></a>GetSingleShape</p>
</td>
<td class="cellrowborder" valign="top" width="81.83%" headers="mcps1.2.3.1.2 "><p id="p16741913104811"><a name="p16741913104811"></a><a name="p16741913104811"></a>获取计算后的singleCoreM/singleCoreN/singleCoreK。</p>
</td>
</tr>
<tr id="row66836102185"><td class="cellrowborder" valign="top" width="18.17%" headers="mcps1.2.3.1.1 "><p id="p3915151921810"><a name="p3915151921810"></a><a name="p3915151921810"></a>SetAlignSplit</p>
</td>
<td class="cellrowborder" valign="top" width="81.83%" headers="mcps1.2.3.1.2 "><p id="p56832010121814"><a name="p56832010121814"></a><a name="p56832010121814"></a>设置多核切分时singleCoreM/singleCoreN/singleCoreK的对齐值</p>
</td>
</tr>
<tr id="row94141412144815"><td class="cellrowborder" valign="top" width="18.17%" headers="mcps1.2.3.1.1 "><p id="p874113135485"><a name="p874113135485"></a><a name="p874113135485"></a>GetCoreNum</p>
</td>
<td class="cellrowborder" valign="top" width="81.83%" headers="mcps1.2.3.1.2 "><p id="p27411313194817"><a name="p27411313194817"></a><a name="p27411313194817"></a>获得多核切分后， 使用的numBlocks。</p>
</td>
</tr>
<tr id="row1395374761814"><td class="cellrowborder" valign="top" width="18.17%" headers="mcps1.2.3.1.1 "><p id="p1095420479183"><a name="p1095420479183"></a><a name="p1095420479183"></a>SetSplitK</p>
</td>
<td class="cellrowborder" valign="top" width="81.83%" headers="mcps1.2.3.1.2 "><p id="p1295454715187"><a name="p1295454715187"></a><a name="p1295454715187"></a>多核场景，使能切K轴。建议使用EnableMultiCoreSplitK接口。</p>
</td>
</tr>
<tr id="row1697272115918"><td class="cellrowborder" valign="top" width="18.17%" headers="mcps1.2.3.1.1 "><p id="p17972322593"><a name="p17972322593"></a><a name="p17972322593"></a>EnableMultiCoreSplitK</p>
</td>
<td class="cellrowborder" valign="top" width="81.83%" headers="mcps1.2.3.1.2 "><p id="p129721922595"><a name="p129721922595"></a><a name="p129721922595"></a>多核场景，使能切K轴。</p>
</td>
</tr>
</tbody>
</table>

**表 3**  BatchMatmulTiling其他接口

<a name="table1225725310207"></a>
<table><thead align="left"><tr id="row1325835312204"><th class="cellrowborder" valign="top" width="18.17%" id="mcps1.2.3.1.1"><p id="p3258175315208"><a name="p3258175315208"></a><a name="p3258175315208"></a>接口</p>
</th>
<th class="cellrowborder" valign="top" width="81.83%" id="mcps1.2.3.1.2"><p id="p11258253112015"><a name="p11258253112015"></a><a name="p11258253112015"></a>功能</p>
</th>
</tr>
</thead>
<tbody><tr id="row15258253162014"><td class="cellrowborder" valign="top" width="18.17%" headers="mcps1.2.3.1.1 "><p id="p9258205314208"><a name="p9258205314208"></a><a name="p9258205314208"></a>GetCoreNum</p>
</td>
<td class="cellrowborder" valign="top" width="81.83%" headers="mcps1.2.3.1.2 "><p id="p192587535208"><a name="p192587535208"></a><a name="p192587535208"></a>获得多核切分后， 使用的numBlocks。</p>
</td>
</tr>
</tbody>
</table>

## 需要包含的头文件<a name="section747122472011"></a>

-   Matmul单核Tiling

    ```
    #include "lib/matmul/matmul_tiling.h"
    ```

-   Matmul多核Tiling

    ```
    #include "lib/matmul/bmm_tiling.h"
    ```

-   BatchMatmul Tiling

    ```
    #include "lib/matmul/bmm_tiling.h"
    ```

