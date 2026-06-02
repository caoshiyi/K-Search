# GetSelectMaxMinTmpSize<a name="ZH-CN_TOPIC_0000002554423535"></a>

## 功能说明<a name="section618mcpsimp"></a>

kernel侧Select接口的计算需要开发者申请临时空间，本接口用于在host侧获取申请的最大最小临时空间大小，开发者基于此范围选择合适的空间大小作为Tiling参数传递到kernel侧使用。

-   为保证功能正确，申请的临时空间大小不能小于最小临时空间大小；
-   在最小临时空间-最大临时空间范围内，随着临时空间增大，kernel侧接口计算性能会有一定程度的优化提升。为了达到更好的性能，开发者可以根据实际的内存使用情况进行空间申请。

## 函数原型<a name="section620mcpsimp"></a>

> **说明：** 
>GetSelectWithBytesMaskMinTmpSize、GetSelectWithBytesMaskMaxTmpSize、GetSelectWithBytesMaskMaxMinTmpSize接口废弃，并将在后续版本移除，请不要使用该接口。请使用GetSelectMinTmpSize、GetSelectMaxTmpSize、GetSelectMaxMinTmpSize接口。

-   获取最小临时空间大小

    ```
    uint32_t GetSelectMinTmpSize(const ge::Shape& src0Shape, const ge::Shape& src1Shape, const uint32_t srcTypeSize, const ge::Shape& maskShape, const uint32_t maskTypeSize, const bool isReuseMask)
    ```

    ```
    uint32_t GetSelectWithBytesMaskMinTmpSize(const ge::Shape& src0Shape, const ge::Shape& src1Shape, const uint32_t srcTypeSize, const ge::Shape& maskShape, const uint32_t maskTypeSize, const bool isReuseMask)
    ```

-   获取最大临时空间大小

    ```
    uint32_t GetSelectMaxTmpSize(const ge::Shape& src0Shape, const ge::Shape& src1Shape, const uint32_t srcTypeSize, const ge::Shape& maskShape, const uint32_t maskTypeSize, const bool isReuseMask)
    ```

    ```
    uint32_t GetSelectWithBytesMaskMaxTmpSize(const ge::Shape& src0Shape, const ge::Shape& src1Shape, const uint32_t srcTypeSize, const ge::Shape& maskShape, const uint32_t maskTypeSize, const bool isReuseMask)
    ```

-   获取最大和最小临时空间大小

    ```
    void GetSelectMaxMinTmpSize(const ge::Shape& src0Shape, const ge::Shape& src1Shape, const uint32_t srcTypeSize, const ge::Shape& maskShape, const uint32_t maskTypeSize, const bool isReuseMask, uint32_t& maxValue, uint32_t& minValue)
    ```

    ```
    void GetSelectWithBytesMaskMaxMinTmpSize(const ge::Shape& src0Shape, const ge::Shape& src1Shape, const uint32_t srcTypeSize, const ge::Shape& maskShape, const uint32_t maskTypeSize, const bool isReuseMask, uint32_t& maxValue, uint32_t& minValue)
    ```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  接口参数列表

<a name="table729818506422"></a>
<table><thead align="left"><tr id="row11299950204217"><th class="cellrowborder" valign="top" width="16.869999999999997%" id="mcps1.2.4.1.1"><p id="p1029955044218"><a name="p1029955044218"></a><a name="p1029955044218"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="12.8%" id="mcps1.2.4.1.2"><p id="p13216203215467"><a name="p13216203215467"></a><a name="p13216203215467"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="70.33%" id="mcps1.2.4.1.3"><p id="p1629911506421"><a name="p1629911506421"></a><a name="p1629911506421"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row12299165018421"><td class="cellrowborder" valign="top" width="16.869999999999997%" headers="mcps1.2.4.1.1 "><p id="p1329915004219"><a name="p1329915004219"></a><a name="p1329915004219"></a>src0Shape</p>
</td>
<td class="cellrowborder" valign="top" width="12.8%" headers="mcps1.2.4.1.2 "><p id="p521618329466"><a name="p521618329466"></a><a name="p521618329466"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="70.33%" headers="mcps1.2.4.1.3 "><p id="p8299155010420"><a name="p8299155010420"></a><a name="p8299155010420"></a>输入src0的shape信息。src0为scalar时，shape应为{1}。</p>
</td>
</tr>
<tr id="row12621149121615"><td class="cellrowborder" valign="top" width="16.869999999999997%" headers="mcps1.2.4.1.1 "><p id="p8543910177"><a name="p8543910177"></a><a name="p8543910177"></a>src1Shape</p>
</td>
<td class="cellrowborder" valign="top" width="12.8%" headers="mcps1.2.4.1.2 "><p id="p754318121712"><a name="p754318121712"></a><a name="p754318121712"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="70.33%" headers="mcps1.2.4.1.3 "><p id="p1754313118172"><a name="p1754313118172"></a><a name="p1754313118172"></a>输入src1的shape信息。src1为scalar时，shape应为{1}。</p>
</td>
</tr>
<tr id="row19299125011422"><td class="cellrowborder" valign="top" width="16.869999999999997%" headers="mcps1.2.4.1.1 "><p id="p1929916502421"><a name="p1929916502421"></a><a name="p1929916502421"></a>srcTypeSize</p>
</td>
<td class="cellrowborder" valign="top" width="12.8%" headers="mcps1.2.4.1.2 "><p id="p172166321461"><a name="p172166321461"></a><a name="p172166321461"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="70.33%" headers="mcps1.2.4.1.3 "><p id="p8299150154218"><a name="p8299150154218"></a><a name="p8299150154218"></a>输入srcTensor的数据类型大小，比如数据类型为half，此处应传入2。</p>
</td>
</tr>
<tr id="row2683814161713"><td class="cellrowborder" valign="top" width="16.869999999999997%" headers="mcps1.2.4.1.1 "><p id="p366593112177"><a name="p366593112177"></a><a name="p366593112177"></a>maskShape</p>
</td>
<td class="cellrowborder" valign="top" width="12.8%" headers="mcps1.2.4.1.2 "><p id="p966515314175"><a name="p966515314175"></a><a name="p966515314175"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="70.33%" headers="mcps1.2.4.1.3 "><p id="p1466515316179"><a name="p1466515316179"></a><a name="p1466515316179"></a>输入maskTensor的shape信息。</p>
</td>
</tr>
<tr id="row329161811710"><td class="cellrowborder" valign="top" width="16.869999999999997%" headers="mcps1.2.4.1.1 "><p id="p1657713270171"><a name="p1657713270171"></a><a name="p1657713270171"></a>maskTypeSize</p>
</td>
<td class="cellrowborder" valign="top" width="12.8%" headers="mcps1.2.4.1.2 "><p id="p105777273171"><a name="p105777273171"></a><a name="p105777273171"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="70.33%" headers="mcps1.2.4.1.3 "><p id="p757710277179"><a name="p757710277179"></a><a name="p757710277179"></a>输入maskTensor的数据类型大小，比如数据类型为bool，此处应传入1。</p>
</td>
</tr>
<tr id="row5299125054217"><td class="cellrowborder" valign="top" width="16.869999999999997%" headers="mcps1.2.4.1.1 "><p id="p9777142884312"><a name="p9777142884312"></a><a name="p9777142884312"></a>isReuseMask</p>
</td>
<td class="cellrowborder" valign="top" width="12.8%" headers="mcps1.2.4.1.2 "><p id="p1221613214466"><a name="p1221613214466"></a><a name="p1221613214466"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="70.33%" headers="mcps1.2.4.1.3 "><p id="p277815284439"><a name="p277815284439"></a><a name="p277815284439"></a>是否复用maskTensor输入的空间。与kernel侧保持一致。</p>
</td>
</tr>
<tr id="row1811014811322"><td class="cellrowborder" valign="top" width="16.869999999999997%" headers="mcps1.2.4.1.1 "><p id="p2029945305"><a name="p2029945305"></a><a name="p2029945305"></a>maxValue</p>
</td>
<td class="cellrowborder" valign="top" width="12.8%" headers="mcps1.2.4.1.2 "><p id="p14298135802"><a name="p14298135802"></a><a name="p14298135802"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="70.33%" headers="mcps1.2.4.1.3 "><p id="p6784154411216"><a name="p6784154411216"></a><a name="p6784154411216"></a>Select接口能完成计算所需最大临时空间大小。</p>
<div class="note" id="note1275197121212"><a name="note1275197121212"></a><a name="note1275197121212"></a><span class="notetitle"> 说明： </span><div class="notebody"><p id="p1040915016493"><a name="p1040915016493"></a><a name="p1040915016493"></a>maxValue仅作为参考值，有可能大于<span id="ph1088254310583"><a name="ph1088254310583"></a><a name="ph1088254310583"></a>Unified Buffer</span>剩余空间的大小，该场景下，开发者需要根据<span id="ph20796650203718"><a name="ph20796650203718"></a><a name="ph20796650203718"></a>Unified Buffer</span>剩余空间的大小来选取合适的临时空间大小。</p>
</div></div>
</td>
</tr>
<tr id="row4321610123220"><td class="cellrowborder" valign="top" width="16.869999999999997%" headers="mcps1.2.4.1.1 "><p id="p13214027173615"><a name="p13214027173615"></a><a name="p13214027173615"></a>minValue</p>
</td>
<td class="cellrowborder" valign="top" width="12.8%" headers="mcps1.2.4.1.2 "><p id="p721432716369"><a name="p721432716369"></a><a name="p721432716369"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="70.33%" headers="mcps1.2.4.1.3 "><p id="p2797134611121"><a name="p2797134611121"></a><a name="p2797134611121"></a>Select接口能完成计算所需最小临时空间大小。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

GetSelectMinTmpSize返回Select接口能完成计算所需最小临时空间大小。

GetSelectMaxTmpSize返回Select接口能完成计算所需最大临时空间大小。

GetSelectMaxMinTmpSize无返回值。

## 约束说明<a name="section92611953111217"></a>

无

## 调用示例<a name="section642mcpsimp"></a>

```
std::vector<int64_t> shape0Vec = {64, 128};
std::vector<int64_t> shape1Vec = {1};
std::vector<int64_t> mask1Vec = {64, 128};
ge::Shape src0Shape(shape0Vec);
ge::Shape src1Shape(shape1Vec);
ge::Shape maskShape(mask1Vec);
uint32_t maxValue = 0;
uint32_t minValue = 0;
AscendC::GetSelectMaxMinTmpSize(src0Shape, src1Shape, 2, maskShape, 1, false, maxValue, minValue);
```

