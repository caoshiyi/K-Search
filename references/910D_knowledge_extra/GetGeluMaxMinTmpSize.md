# GetGeluMaxMinTmpSize<a name="ZH-CN_TOPIC_0000002554423617"></a>

## 功能说明<a name="section618mcpsimp"></a>

kernel侧Gelu、FasterGelu、FasterGeluV2接口的计算需要开发者预留/申请临时空间，本接口用于在host侧获取预留/申请的最大最小临时空间大小，开发者基于此范围选择合适的空间大小作为Tiling参数传递到kernel侧使用。

-   为保证功能正确，预留/申请的临时空间大小不能小于最小临时空间大小；
-   在最小临时空间-最大临时空间范围内，随着临时空间增大，kernel侧接口计算性能会有一定程度的优化提升。为了达到更好的性能，开发者可以根据实际的内存使用情况进行空间预留/申请。

## 函数原型<a name="section620mcpsimp"></a>

-   获取Kernel接口计算所需最大临时空间大小的接口

    ```
    uint32_t GetGeluMaxTmpSize(const ge::Shape& srcShape, const uint32_t typeSize)
    ```

-   获取Kernel接口计算所需最小临时空间大小

    ```
    uint32_t GetGeluMinTmpSize(const ge::Shape& srcShape, const uint32_t typeSize)
    ```

-   获取Kernel接口计算所需最大/最小临时空间的接口

    ```
    void GetGeluMaxMinTmpSize(const ge::Shape& srcShape, const uint32_t typeSize, uint32_t& maxValue, uint32_t& minValue)
    ```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  参数列表

<a name="table729818506422"></a>
<table><thead align="left"><tr id="row11299950204217"><th class="cellrowborder" valign="top" width="16.89%" id="mcps1.2.4.1.1"><p id="p1029955044218"><a name="p1029955044218"></a><a name="p1029955044218"></a>接口</p>
</th>
<th class="cellrowborder" valign="top" width="7.5200000000000005%" id="mcps1.2.4.1.2"><p id="p13216203215467"><a name="p13216203215467"></a><a name="p13216203215467"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="75.59%" id="mcps1.2.4.1.3"><p id="p1629911506421"><a name="p1629911506421"></a><a name="p1629911506421"></a>功能</p>
</th>
</tr>
</thead>
<tbody><tr id="row12299165018421"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p1329915004219"><a name="p1329915004219"></a><a name="p1329915004219"></a>srcShape</p>
</td>
<td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.2.4.1.2 "><p id="p521618329466"><a name="p521618329466"></a><a name="p521618329466"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.59%" headers="mcps1.2.4.1.3 "><p id="p8299155010420"><a name="p8299155010420"></a><a name="p8299155010420"></a>输入的shape信息。</p>
</td>
</tr>
<tr id="row19299125011422"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p1929916502421"><a name="p1929916502421"></a><a name="p1929916502421"></a>typeSize</p>
</td>
<td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.2.4.1.2 "><p id="p172166321461"><a name="p172166321461"></a><a name="p172166321461"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.59%" headers="mcps1.2.4.1.3 "><p id="p8299150154218"><a name="p8299150154218"></a><a name="p8299150154218"></a>输入的数据类型大小，单位为字节。比如输入的数据类型为half，此处应传入2。</p>
</td>
</tr>
<tr id="row6777152811436"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p23791451102416"><a name="p23791451102416"></a><a name="p23791451102416"></a>maxValue</p>
</td>
<td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.2.4.1.2 "><p id="p470071112510"><a name="p470071112510"></a><a name="p470071112510"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="75.59%" headers="mcps1.2.4.1.3 "><p id="p1890195216531"><a name="p1890195216531"></a><a name="p1890195216531"></a>Gelu、FasterGelu、FasterGeluV2接口能完成计算所需的最大临时空间大小，超出该值的空间不会被该接口使用。在最小临时空间-最大临时空间范围内，随着临时空间增大，kernel侧接口计算性能会有一定程度的优化提升。为了达到更好的性能，开发者可以根据实际的内存使用情况进行空间预留/申请。最大空间大小为0表示计算不需要临时空间。</p>
<div class="note" id="note1275197121212"><a name="note1275197121212"></a><a name="note1275197121212"></a><span class="notetitle"> 说明： </span><div class="notebody"><p id="p1040915016493"><a name="p1040915016493"></a><a name="p1040915016493"></a>maxValue仅作为参考值，有可能大于<span id="ph1088254310583"><a name="ph1088254310583"></a><a name="ph1088254310583"></a>Unified Buffer</span>剩余空间的大小，该场景下，开发者需要根据<span id="ph20796650203718"><a name="ph20796650203718"></a><a name="ph20796650203718"></a>Unified Buffer</span>剩余空间的大小来选取合适的临时空间大小。</p>
</div></div>
</td>
</tr>
<tr id="row6563634154317"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p956314345431"><a name="p956314345431"></a><a name="p956314345431"></a>minValue</p>
</td>
<td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.2.4.1.2 "><p id="p42161432144610"><a name="p42161432144610"></a><a name="p42161432144610"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="75.59%" headers="mcps1.2.4.1.3 "><p id="p66955018287"><a name="p66955018287"></a><a name="p66955018287"></a>Gelu、FasterGelu、FasterGeluV2接口能完成计算所需最小临时空间大小。为保证功能正确，接口计算时预留/申请的临时空间不能小于该数值。最小空间大小为0表示计算不需要临时空间。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

**GetGeluMaxTmpSize**返回kernel侧接口能完成计算所需最大临时空间大小。

**GetGeluMinTmpSize**返回kernel侧接口能完成计算所需最小临时空间大小。

**GetGeluMaxMinTmpSize**无返回值。

## 约束说明<a name="section92611953111217"></a>

无

## 调用示例<a name="section642mcpsimp"></a>

```
// 输入shape信息为1024;算子输入的数据类型为half;
std::vector<int64_t> shape_vec = {1024};
ge::Shape srcShape(shape_vec);
uint32_t typeSize = 2;
uint32_t maxValue = 0;
uint32_t minValue = 0;
AscendC::GetGeluMaxMinTmpSize(srcShape, typeSize, maxValue, minValue);
```

