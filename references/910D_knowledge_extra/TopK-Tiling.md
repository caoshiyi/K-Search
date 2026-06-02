# TopK Tiling<a name="ZH-CN_TOPIC_0000002523304892"></a>

## 功能说明<a name="section10757913172313"></a>

用于获取TopK Tiling参数。

Ascend C提供TopK Tiling API，方便用户获取TopK kernel计算时所需的Tiling参数。阅读本节之前，请先参考[Tiling实现](Host侧Tiling实现.md)了解Tiling实现基本流程。

获取Tiling参数主要分为如下两步：

1.  获取TopK接口计算所需最小和最大临时空间大小，注意该步骤不是必须的，只是作为一个参考，供合理分配计算空间。
2.  获取TopK kernel侧接口所需tiling参数。

    TopK Tiling结构体的定义如下，开发者无需关注该tiling结构的具体信息，只需要传递到kernel侧，传入TopK高阶API接口，直接进行使用即可。

    ```
    struct TopkTiling {
        int32_t tmpLocalSize = 0;
        int32_t allDataSize = 0;
        int32_t innerDataSize = 0;
        uint32_t sortRepeat = 0;
        int32_t mrgSortRepeat = 0;
        int32_t kAlignFourBytes = 0;
        int32_t kAlignTwoBytes = 0;
        int32_t maskOffset = 0;
        int32_t maskVreducev2FourBytes = 0;
        int32_t maskVreducev2TwoBytes = 0;
        int32_t mrgSortSrc1offset = 0;
        int32_t mrgSortSrc2offset = 0;
        int32_t mrgSortSrc3offset = 0;
        int32_t mrgSortTwoQueueSrc1Offset = 0;
        int32_t mrgFourQueueTailPara1 = 0;
        int32_t mrgFourQueueTailPara2 = 0;
        int32_t srcIndexOffset = 0;
        uint32_t copyUbToUbBlockCount = 0;
        int32_t topkMrgSrc1MaskSizeOffset = 0;
        int32_t topkNSmallSrcIndexOffset = 0;
        uint32_t vreduceValMask0 = 0;
        uint32_t vreduceValMask1 = 0;
        uint32_t vreduceIdxMask0 = 0;
        uint32_t vreduceIdxMask1 = 0;
        uint16_t vreducehalfValMask0 = 0;
        uint16_t vreducehalfValMask1 = 0;
        uint16_t vreducehalfValMask2 = 0;
        uint16_t vreducehalfValMask3 = 0;
        uint16_t vreducehalfValMask4 = 0;
        uint16_t vreducehalfValMask5 = 0;
        uint16_t vreducehalfValMask6 = 0;
        uint16_t vreducehalfValMask7 = 0;    
    };
    ```

## 函数原型<a name="section11959319102316"></a>

```
bool GetTopKMaxMinTmpSize(const platform_ascendc::PlatformAscendC& ascendcPlatform, const int32_t inner, const int32_t outter, const bool isReuseSource, const bool isInitIndex, enum TopKMode mode, const bool isLargest, const uint32_t dataTypeSize, uint32_t& maxValue, uint32_t& minValue)
```

```
bool GetTopKMaxMinTmpSize(const int32_t inner, const int32_t outter, const int32_t k, const bool isReuseSource, const bool isInitIndex, enum TopKMode mode, const bool isLargest, ge::DataType dataType, const TopKConfig& config, uint32_t& maxValue, uint32_t& minValue)
```

```
bool TopKTilingFunc(const platform_ascendc::PlatformAscendC& ascendcPlatform, const int32_t inner, const int32_t outter, const int32_t k, const uint32_t dataTypeSize, const bool isInitIndex, enum TopKMode mode, const bool isLargest, optiling::TopkTiling& topKTiling)
```

```
bool TopKTilingFunc(const platform_ascendc::PlatformAscendC& ascendcPlatform, const int32_t inner, const int32_t outter, const int32_t k, const uint32_t dataTypeSize, const bool isInitIndex, enum TopKMode mode, const bool isLargest, AscendC::tiling::TopkTiling& topKTiling)
```

## 参数说明<a name="section7883162615235"></a>

**表 1**  GetTopKMaxMinTmpSize接口参数列表

<a name="table171406364408"></a>
<table><thead align="left"><tr id="row21406365408"><th class="cellrowborder" valign="top" width="16.869999999999997%" id="mcps1.2.4.1.1"><p id="p17140836184013"><a name="p17140836184013"></a><a name="p17140836184013"></a>接口</p>
</th>
<th class="cellrowborder" valign="top" width="10.63%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="72.5%" id="mcps1.2.4.1.3"><p id="p141405369409"><a name="p141405369409"></a><a name="p141405369409"></a>功能</p>
</th>
</tr>
</thead>
<tbody><tr id="row84142034449"><td class="cellrowborder" valign="top" width="16.869999999999997%" headers="mcps1.2.4.1.1 "><p id="p171948581597"><a name="p171948581597"></a><a name="p171948581597"></a>ascendcPlatform</p>
</td>
<td class="cellrowborder" valign="top" width="10.63%" headers="mcps1.2.4.1.2 "><p id="p73181356192211"><a name="p73181356192211"></a><a name="p73181356192211"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.5%" headers="mcps1.2.4.1.3 "><p id="p340161012618"><a name="p340161012618"></a><a name="p340161012618"></a>传入硬件平台的信息，PlatformAscendC定义请参见<a href="构造及析构函数.md">构造及析构函数</a>。</p>
</td>
</tr>
<tr id="row31417367406"><td class="cellrowborder" valign="top" width="16.869999999999997%" headers="mcps1.2.4.1.1 "><p id="p12604203117452"><a name="p12604203117452"></a><a name="p12604203117452"></a>inner</p>
</td>
<td class="cellrowborder" valign="top" width="10.63%" headers="mcps1.2.4.1.2 "><p id="p186293346150"><a name="p186293346150"></a><a name="p186293346150"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.5%" headers="mcps1.2.4.1.3 "><p id="p14604183114514"><a name="p14604183114514"></a><a name="p14604183114514"></a>表示TopK接口输入srcLocal的内轴长度，该参数的取值为32的整数倍。</p>
</td>
</tr>
<tr id="row20852161535115"><td class="cellrowborder" valign="top" width="16.869999999999997%" headers="mcps1.2.4.1.1 "><p id="p17852915205118"><a name="p17852915205118"></a><a name="p17852915205118"></a>outter</p>
</td>
<td class="cellrowborder" valign="top" width="10.63%" headers="mcps1.2.4.1.2 "><p id="p12852315175111"><a name="p12852315175111"></a><a name="p12852315175111"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.5%" headers="mcps1.2.4.1.3 "><p id="p15852015175111"><a name="p15852015175111"></a><a name="p15852015175111"></a>表示TopK接口输入srcLocal的外轴长度。</p>
</td>
</tr>
<tr id="row15274173155912"><td class="cellrowborder" valign="top" width="16.869999999999997%" headers="mcps1.2.4.1.1 "><p id="p995153814592"><a name="p995153814592"></a><a name="p995153814592"></a>k</p>
</td>
<td class="cellrowborder" valign="top" width="10.63%" headers="mcps1.2.4.1.2 "><p id="p139518389595"><a name="p139518389595"></a><a name="p139518389595"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.5%" headers="mcps1.2.4.1.3 "><p id="p1895183825919"><a name="p1895183825919"></a><a name="p1895183825919"></a>获取前k个最大值或最小值及其对应的索引。</p>
</td>
</tr>
<tr id="row2725181895118"><td class="cellrowborder" valign="top" width="16.869999999999997%" headers="mcps1.2.4.1.1 "><p id="p103833446514"><a name="p103833446514"></a><a name="p103833446514"></a>isReuseSource</p>
</td>
<td class="cellrowborder" valign="top" width="10.63%" headers="mcps1.2.4.1.2 "><p id="p872519182513"><a name="p872519182513"></a><a name="p872519182513"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.5%" headers="mcps1.2.4.1.3 "><p id="p185124380135"><a name="p185124380135"></a><a name="p185124380135"></a>中间变量是否能够复用输入内存。与kernel侧接口的isReuseSrc保持一致。</p>
</td>
</tr>
<tr id="row18245182114517"><td class="cellrowborder" valign="top" width="16.869999999999997%" headers="mcps1.2.4.1.1 "><p id="p10245102105116"><a name="p10245102105116"></a><a name="p10245102105116"></a>isInitIndex</p>
</td>
<td class="cellrowborder" valign="top" width="10.63%" headers="mcps1.2.4.1.2 "><p id="p1724552115115"><a name="p1724552115115"></a><a name="p1724552115115"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.5%" headers="mcps1.2.4.1.3 "><p id="p1624510219513"><a name="p1624510219513"></a><a name="p1624510219513"></a>是否传入输入数据对应的索引，与kernel侧接口一致。</p>
</td>
</tr>
<tr id="row179738248519"><td class="cellrowborder" valign="top" width="16.869999999999997%" headers="mcps1.2.4.1.1 "><p id="p18973162405110"><a name="p18973162405110"></a><a name="p18973162405110"></a>mode</p>
</td>
<td class="cellrowborder" valign="top" width="10.63%" headers="mcps1.2.4.1.2 "><p id="p1897492417517"><a name="p1897492417517"></a><a name="p1897492417517"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.5%" headers="mcps1.2.4.1.3 "><p id="p4974112412513"><a name="p4974112412513"></a><a name="p4974112412513"></a>选择TopKMode::TOPK_NORMAL模式或者TopKMode::TOPK_NSMALL模式，与kernel侧接口一致。</p>
</td>
</tr>
<tr id="row485814176529"><td class="cellrowborder" valign="top" width="16.869999999999997%" headers="mcps1.2.4.1.1 "><p id="p1785811745212"><a name="p1785811745212"></a><a name="p1785811745212"></a>isLargest</p>
</td>
<td class="cellrowborder" valign="top" width="10.63%" headers="mcps1.2.4.1.2 "><p id="p198588170521"><a name="p198588170521"></a><a name="p198588170521"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.5%" headers="mcps1.2.4.1.3 "><p id="p585831725214"><a name="p585831725214"></a><a name="p585831725214"></a>表示降序/升序，true表示降序，false表示升序。与kernel侧接口一致。</p>
</td>
</tr>
<tr id="row106551243010"><td class="cellrowborder" valign="top" width="16.869999999999997%" headers="mcps1.2.4.1.1 "><p id="p1865594506"><a name="p1865594506"></a><a name="p1865594506"></a>dataType</p>
</td>
<td class="cellrowborder" valign="top" width="10.63%" headers="mcps1.2.4.1.2 "><p id="p1765674105"><a name="p1765674105"></a><a name="p1765674105"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.5%" headers="mcps1.2.4.1.3 "><p id="p206561415018"><a name="p206561415018"></a><a name="p206561415018"></a>表示待排序数据的数据类型。该参数的取值与Kernel接口参数srcLocal的数据类型保持一致。</p>
</td>
</tr>
<tr id="row69871728191512"><td class="cellrowborder" valign="top" width="16.869999999999997%" headers="mcps1.2.4.1.1 "><p id="p1498802818152"><a name="p1498802818152"></a><a name="p1498802818152"></a>config</p>
</td>
<td class="cellrowborder" valign="top" width="10.63%" headers="mcps1.2.4.1.2 "><p id="p199881628191512"><a name="p199881628191512"></a><a name="p199881628191512"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.5%" headers="mcps1.2.4.1.3 "><p id="p4979658192614"><a name="p4979658192614"></a><a name="p4979658192614"></a>TopK计算的相关配置，包括算法选择、取最大值或最小值、是否对结果排序。该参数的配置需要与TopK Kernel接口模板参数的配置保持一致。</p>
<a name="ul1239918433297"></a><a name="ul1239918433297"></a><ul id="ul1239918433297"><li>algo：选择的排序算法。默认为MERGE_SORT算法，当前仅支持RADIX_SELECT算法，用户需要显式指定algo为TopKAlgo::RADIX_SELECT。</li><li>order：表示获取前k个最大值或者获取前k个最小值，取值如下：<a name="ul425516518113"></a><a name="ul425516518113"></a><ul id="ul425516518113"><li>UNSET：默认值，按照函数参数isLargest的配置实现。isLargest为true时，取前k个最大值及其对应的索引，isLargest为false，取前k个最小值及其对应的索引。</li><li>LARGEST：表示取前k个最大值及其对应的索引。取值为LARGEST时，函数参数isLargest的配置不生效。</li><li>SMALLEST：表示取前k个最小值及其对应的索引。取值为SMALLEST时，函数参数isLargest的配置不生效。</li></ul>
</li><li>sorted：表示是否对输出结果进行排序。取值为true，对输出结果进行排序；取值为false，不对输出结果进行排序。</li></ul>
<a name="screen1563331371317"></a><a name="screen1563331371317"></a><pre class="screen" codetype="Cpp" id="screen1563331371317">struct TopKConfig {
    TopKAlgo algo = TopKAlgo::MERGE_SORT;
    TopKOrder order = TopKOrder::UNSET;
    bool sorted = true;
};
enum class TopKAlgo {
    RADIX_SELECT,
    MERGE_SORT
};
enum class TopKOrder {
    UNSET,
    LARGEST,
    SMALLEST
};</pre>
</td>
</tr>
<tr id="row747183111226"><td class="cellrowborder" valign="top" width="16.869999999999997%" headers="mcps1.2.4.1.1 "><p id="p17472103152216"><a name="p17472103152216"></a><a name="p17472103152216"></a>dataTypeSize</p>
</td>
<td class="cellrowborder" valign="top" width="10.63%" headers="mcps1.2.4.1.2 "><p id="p154721631152217"><a name="p154721631152217"></a><a name="p154721631152217"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.5%" headers="mcps1.2.4.1.3 "><p id="p1247203182211"><a name="p1247203182211"></a><a name="p1247203182211"></a>参与计算的srcLocal数据类型的大小，比如half=2， float=4</p>
</td>
</tr>
<tr id="row3317756162211"><td class="cellrowborder" valign="top" width="16.869999999999997%" headers="mcps1.2.4.1.1 "><p id="p0605141113473"><a name="p0605141113473"></a><a name="p0605141113473"></a>maxValue</p>
</td>
<td class="cellrowborder" valign="top" width="10.63%" headers="mcps1.2.4.1.2 "><p id="p156059117476"><a name="p156059117476"></a><a name="p156059117476"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="72.5%" headers="mcps1.2.4.1.3 "><p id="p147373925711"><a name="p147373925711"></a><a name="p147373925711"></a>TopK接口内部完成计算需要的最大临时空间大小，单位是Byte。</p>
<div class="note" id="note1275197121212"><a name="note1275197121212"></a><a name="note1275197121212"></a><span class="notetitle"> 说明： </span><div class="notebody"><p id="p1040915016493"><a name="p1040915016493"></a><a name="p1040915016493"></a>maxValue仅作为参考值，有可能大于<span id="ph1088254310583"><a name="ph1088254310583"></a><a name="ph1088254310583"></a>Unified Buffer</span>剩余空间的大小，该场景下，开发者需要根据<span id="ph20796650203718"><a name="ph20796650203718"></a><a name="ph20796650203718"></a>Unified Buffer</span>剩余空间的大小来选取合适的临时空间大小。</p>
</div></div>
</td>
</tr>
<tr id="row189513915513"><td class="cellrowborder" valign="top" width="16.869999999999997%" headers="mcps1.2.4.1.1 "><p id="p126931621205219"><a name="p126931621205219"></a><a name="p126931621205219"></a>minValue</p>
</td>
<td class="cellrowborder" valign="top" width="10.63%" headers="mcps1.2.4.1.2 "><p id="p4693142135214"><a name="p4693142135214"></a><a name="p4693142135214"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="72.5%" headers="mcps1.2.4.1.3 "><p id="p9693112118529"><a name="p9693112118529"></a><a name="p9693112118529"></a>TopK接口内部完成计算需要的最小临时空间大小，单位是Byte。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  TopKTilingFunc接口参数列表

<a name="table729818506422"></a>
<table><thead align="left"><tr id="row11299950204217"><th class="cellrowborder" valign="top" width="16.89%" id="mcps1.2.4.1.1"><p id="p1029955044218"><a name="p1029955044218"></a><a name="p1029955044218"></a>接口</p>
</th>
<th class="cellrowborder" valign="top" width="10.4%" id="mcps1.2.4.1.2"><p id="p13216203215467"><a name="p13216203215467"></a><a name="p13216203215467"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="72.71%" id="mcps1.2.4.1.3"><p id="p1629911506421"><a name="p1629911506421"></a><a name="p1629911506421"></a>功能</p>
</th>
</tr>
</thead>
<tbody><tr id="row29912279513"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p6651332355"><a name="p6651332355"></a><a name="p6651332355"></a>ascendcPlatform</p>
</td>
<td class="cellrowborder" valign="top" width="10.4%" headers="mcps1.2.4.1.2 "><p id="p1765115321057"><a name="p1765115321057"></a><a name="p1765115321057"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.71%" headers="mcps1.2.4.1.3 "><p id="p1465118324515"><a name="p1465118324515"></a><a name="p1465118324515"></a>传入硬件平台的信息，PlatformAscendC定义请参见<a href="构造及析构函数.md">构造及析构函数</a>。</p>
</td>
</tr>
<tr id="row2406423535"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p1239662315320"><a name="p1239662315320"></a><a name="p1239662315320"></a>inner</p>
</td>
<td class="cellrowborder" valign="top" width="10.4%" headers="mcps1.2.4.1.2 "><p id="p14396192316310"><a name="p14396192316310"></a><a name="p14396192316310"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.71%" headers="mcps1.2.4.1.3 "><p id="p19396142311310"><a name="p19396142311310"></a><a name="p19396142311310"></a>表示TopK接口输入srcLocal的内轴长度，该参数的取值为32的整数倍。</p>
</td>
</tr>
<tr id="row5406132317319"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p13396162313313"><a name="p13396162313313"></a><a name="p13396162313313"></a>outter</p>
</td>
<td class="cellrowborder" valign="top" width="10.4%" headers="mcps1.2.4.1.2 "><p id="p1339617231930"><a name="p1339617231930"></a><a name="p1339617231930"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.71%" headers="mcps1.2.4.1.3 "><p id="p2039618231936"><a name="p2039618231936"></a><a name="p2039618231936"></a>表示TopK接口输入srcLocal的外轴长度。</p>
</td>
</tr>
<tr id="row26411269918"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p1864192615911"><a name="p1864192615911"></a><a name="p1864192615911"></a>k</p>
</td>
<td class="cellrowborder" valign="top" width="10.4%" headers="mcps1.2.4.1.2 "><p id="p146412611910"><a name="p146412611910"></a><a name="p146412611910"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.71%" headers="mcps1.2.4.1.3 "><p id="p1852917382617"><a name="p1852917382617"></a><a name="p1852917382617"></a>获取前k个最大值或最小值及其对应的索引。</p>
</td>
</tr>
<tr id="row67277811018"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p1929916502421"><a name="p1929916502421"></a><a name="p1929916502421"></a>dataTypeSize</p>
</td>
<td class="cellrowborder" valign="top" width="10.4%" headers="mcps1.2.4.1.2 "><p id="p172166321461"><a name="p172166321461"></a><a name="p172166321461"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.71%" headers="mcps1.2.4.1.3 "><p id="p8299150154218"><a name="p8299150154218"></a><a name="p8299150154218"></a>参与计算的srcLocal数据类型的大小，比如half=2， float=4。</p>
</td>
</tr>
<tr id="row14066231934"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p43963233315"><a name="p43963233315"></a><a name="p43963233315"></a>isInitIndex</p>
</td>
<td class="cellrowborder" valign="top" width="10.4%" headers="mcps1.2.4.1.2 "><p id="p53971823537"><a name="p53971823537"></a><a name="p53971823537"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.71%" headers="mcps1.2.4.1.3 "><p id="p1939712314311"><a name="p1939712314311"></a><a name="p1939712314311"></a>是否传入输入数据对应的索引，与kernel侧接口一致。</p>
</td>
</tr>
<tr id="row940522311315"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p8397723132"><a name="p8397723132"></a><a name="p8397723132"></a>mode</p>
</td>
<td class="cellrowborder" valign="top" width="10.4%" headers="mcps1.2.4.1.2 "><p id="p1839711234320"><a name="p1839711234320"></a><a name="p1839711234320"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.71%" headers="mcps1.2.4.1.3 "><p id="p123979239316"><a name="p123979239316"></a><a name="p123979239316"></a>选择TopKMode::TOPK_NORMAL模式或者TopKMode::TOPK_NSMALL模式，与kernel侧接口一致。</p>
</td>
</tr>
<tr id="row15405123633"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p1239717231739"><a name="p1239717231739"></a><a name="p1239717231739"></a>isLargest</p>
</td>
<td class="cellrowborder" valign="top" width="10.4%" headers="mcps1.2.4.1.2 "><p id="p16397323731"><a name="p16397323731"></a><a name="p16397323731"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.71%" headers="mcps1.2.4.1.3 "><p id="p14397192317310"><a name="p14397192317310"></a><a name="p14397192317310"></a>表示降序/升序，true表示降序，false表示升序。与kernel侧接口一致。</p>
</td>
</tr>
<tr id="row1997124719511"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p174535013513"><a name="p174535013513"></a><a name="p174535013513"></a>topKTiling</p>
</td>
<td class="cellrowborder" valign="top" width="10.4%" headers="mcps1.2.4.1.2 "><p id="p187459509519"><a name="p187459509519"></a><a name="p187459509519"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="72.71%" headers="mcps1.2.4.1.3 "><p id="p174511507514"><a name="p174511507514"></a><a name="p174511507514"></a>输出TopK接口所需的tiling信息。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section148125592413"></a>

GetTopKMaxMinTmpSize返回值为true/false，true表示成功拿到TopK接口内部计算需要的最大和最小临时空间大小；false表示获取失败。

TopKTilingFunc返回值为true/false，true表示成功拿到TopK的Tiling各项参数值；false表示获取失败。

## 约束说明<a name="section92611953111217"></a>

无

## 调用示例<a name="section9931244152418"></a>

如下样例介绍了使用TopK高阶API时host侧获取Tiling参数的流程以及该参数如何在kernel侧使用。

1.  将TopK Tiling结构体参数增加至TilingData结构体，作为TilingData结构体的一个字段。

    ```
    namespace optiling {
    BEGIN_TILING_DATA_DEF(TilingData)
        TILING_DATA_FIELD_DEF(uint32_t, totalLength);
        TILING_DATA_FIELD_DEF(uint32_t, tilenum);
        //添加其他tiling字段
        ...                                
        TILING_DATA_FIELD_DEF(int32_t, k);
        TILING_DATA_FIELD_DEF(bool, islargest);
        TILING_DATA_FIELD_DEF(bool, isinitindex);
        TILING_DATA_FIELD_DEF(bool, ishasfinish);
        TILING_DATA_FIELD_DEF(uint32_t, tmpsize);
        TILING_DATA_FIELD_DEF(int32_t, outter);
        TILING_DATA_FIELD_DEF(int32_t, inner);
        TILING_DATA_FIELD_DEF(int32_t, n);
        TILING_DATA_FIELD_DEF_STRUCT(TopkTiling, topkTilingData);
    END_TILING_DATA_DEF;
    REGISTER_TILING_DATA_CLASS(TopkCustom, TilingData)
    }
    ```

2.  Tiling实现函数中，首先调用GetTopKMaxMinTmpSize接口获取TopK接口能完成计算所需最大/最小临时空间大小，根据该范围结合实际的内存使用情况设置合适的空间大小；然后根据输入shape等信息获取TopK kernel侧接口所需tiling参数。MERGE\_SORT算法参考如下调用示例。

    ```
    namespace optiling {
    const uint32_t NUM_BLOCKS = 8;
    const uint32_t TILE_NUM = 8;
    const int32_t OUTTER = 2;
    const int32_t INNER = 32;
    const int32_t N = 32;
    const int32_t K = 8;
    const bool IS_LARGEST = true;
    const bool IS_INITINDEX = true;
    const bool IS_REUSESOURCE = false;
    static ge::graphStatus TilingFunc(gert::TilingContext* context)
    {
        TilingData tiling;
        uint32_t totalLength = context->GetInputTensor(0)->GetShapeSize();
        context->SetBlockDim(NUM_BLOCKS);
        tiling.set_totalLength(totalLength);
        tiling.set_tileNum(TILE_NUM);
        tiling.set_k(K);
        tiling.set_outter(OUTTER);
        tiling.set_inner(INNER);
        tiling.set_n(N);
        tiling.set_islargest(IS_LARGEST);
        tiling.set_isinitindex(IS_INITINDEX);
        // 设置其他Tiling参数
        ... 
        // 本样例中仅作为样例说明，通过GetTopKMaxMinTmpSize获取最小值并传入，来保证功能正确，开发者可以根据需要传入合适的空间大小。
        uint32_t maxsize = 0;
        uint32_t minsize = 0;
        uint32_t dtypesize = 4;  // float类型
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        AscendC::TopKTilingFunc(ascendcPlatform, tiling.inner, tiling.outter, tiling.k, dtypesize, tiling.isinitindex, AscendC::TopKMode::TOPK_NSMALL, tiling.islargest, tiling.topkTilingData);
        AscendC::GetTopKMaxMinTmpSize(ascendcPlatform, tiling.inner, tiling.outter, IS_REUSESOURCE, tiling.isinitindex, AscendC::TopKMode::TOPK_NSMALL, tiling.islargest, dtypesize, maxsize, minsize);
        tiling.set_tmpsize(minsize);
         ... // 其他逻辑
        tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
    } // namespace optiling
    ```

    RADIX\_SELECT算法参考如下调用示例。

    ```
    namespace optiling
    {
        static ge::graphStatus TilingFunc(gert::TilingContext *context)
        {
            std::map<ge::DataType, uint32_t> dtypeSizes = {
                {ge::DataType::DT_UINT32, 4},
                {ge::DataType::DT_INT32, 4}
            };
            RadixtopkCustomTilingData tiling;
            const gert::RuntimeAttrs *attrs = context->GetAttrs();
            const uint32_t is_init_index = *(attrs->GetAttrPointer<uint32_t>(0));
            const uint32_t is_reuse_src = *(attrs->GetAttrPointer<uint32_t>(1));
            const uint32_t order = *(attrs->GetAttrPointer<uint32_t>(2));
            const uint32_t is_largest = *(attrs->GetAttrPointer<uint32_t>(3));
            const uint32_t outter = *(attrs->GetAttrPointer<uint32_t>(4));
            const uint32_t inner = *(attrs->GetAttrPointer<uint32_t>(5));
            const uint32_t n = *(attrs->GetAttrPointer<uint32_t>(6));
            const uint32_t k = *(attrs->GetAttrPointer<uint32_t>(7));
            const uint32_t k_pad = *(attrs->GetAttrPointer<uint32_t>(8));
            const uint32_t sorted = *(attrs->GetAttrPointer<uint32_t>(9));
            const uint32_t top_mode = *(attrs->GetAttrPointer<uint32_t>(10));
    
            auto xDType = context->GetInputTensor(0)->GetDataType();
            uint32_t typeSize = dtypeSizes.at(xDType);
            AscendC::TopKConfig config;
            config.algo = AscendC::TopKAlgo::RADIX_SELECT;
            if (order == 1) {
                config.order = AscendC::TopKOrder::LARGEST;
            } else if (order == 2) {
                config.order = AscendC::TopKOrder::SMALLEST;
            } else {
                config.order = AscendC::TopKOrder::UNSET;
            }
            if (sorted == 0) {
                config.sorted = false;
            } else {
                config.sorted = true;
            }
            uint32_t maxValue = 0;
            uint32_t minValue = 0;
     
            if (top_mode == 0) {
                AscendC::GetTopKMaxMinTmpSize(inner, outter, k, is_reuse_src, is_init_index, AscendC::TopKMode::TOPK_NORMAL, is_largest, xDType, config, maxValue, minValue);
                context->SetTilingKey(0);
            } else {
                AscendC::GetTopKMaxMinTmpSize(inner, outter, k, is_reuse_src, is_init_index, AscendC::TopKMode::TOPK_NSMALL, is_largest, xDType, config, maxValue, minValue);
                context->SetTilingKey(1);
            }
            context->SetBlockDim(1);
            tiling.set_is_init_index(is_init_index);
            tiling.set_is_reuse_src(is_reuse_src);
            tiling.set_order(order);
            tiling.set_is_largest(is_largest);
            tiling.set_outter(outter);
            tiling.set_inner(inner);
            tiling.set_n(n);
            tiling.set_k(k);
            tiling.set_k_pad(k_pad);
            tiling.set_sorted(sorted);
            tiling.set_top_mode(top_mode);
            tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
            context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
            size_t *currentWorkspace = context->GetWorkspaceSizes(1);
            currentWorkspace[0] = 0;
            return ge::GRAPH_SUCCESS;
        }
    } // namespace optiling
    ```

3.  对应的kernel侧通过在核函数中调用GET\_TILING\_DATA获取TilingData，继而将TilingData中的TopK Tiling信息传入TopK接口参与计算。完整的kernel侧样例请参考[调用示例](TopK.md#section94691236101419)。

    ```
    extern "C" __global__ __aicore__ void topk_custom(GM_ADDR srcVal, GM_ADDR srcIdx, GM_ADDR finishLocal, GM_ADDR dstVal, GM_ADDR dstIdx, GM_ADDR tiling)
    {
        GET_TILING_DATA(tilingData, tiling);
        KernelTopK<float, true, true, false, false, AscendC::TopKMode::TOPK_NSMALL> op;
        op.Init(srcVal, srcIdx, finishLocal, dstVal, dstIdx, tilingData.k, tilingData.islargest, tilingData.tmpsize, tilingData.outter, tilingData.inner, tilingData.n,tilingData.topkTilingData);
        op.Process();
    }
    ```

