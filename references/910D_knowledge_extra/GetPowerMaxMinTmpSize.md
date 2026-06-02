# GetPowerMaxMinTmpSize<a name="ZH-CN_TOPIC_0000002554344523"></a>

## 功能说明<a name="section618mcpsimp"></a>

kernel侧Power接口的计算需要开发者预留/申请临时空间，本接口用于在host侧获取预留/申请的最大和最小临时空间大小，开发者基于此范围选择合适的空间大小作为Tiling参数传递到kernel侧使用。

-   为保证功能正确，预留/申请的临时空间大小不能小于最小临时空间大小；
-   在最小临时空间-最大临时空间范围内，随着临时空间增大，kernel侧接口计算性能会有一定程度的优化提升。为了达到更好的性能，开发者可以根据实际的内存使用情况进行空间预留/申请。

接口内部根据srcShape1、srcShape2输入判断接口为Power\(dstTensor, srcTensor1, srcTensor2\)、Power\(dstTensor, srcTensor1, scalarValue\) 或Power\(dstTensor, scalarValue, srcTensor2\)类型中的哪一种，进而返回对应临时空间大小。

## 函数原型<a name="section620mcpsimp"></a>

```
void GetPowerMaxMinTmpSize(const ge::Shape& srcShape1, const ge::Shape& srcShape2, const bool typeIsInt, const uint32_t typeSize, const bool isReuseSource, uint32_t& maxValue, uint32_t& minValue)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  接口参数列表

<a name="table16492526102317"></a>
<table><thead align="left"><tr id="row1349217266238"><th class="cellrowborder" valign="top" width="16.89%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="10.23%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="72.88%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row6492192612318"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p189045212538"><a name="p189045212538"></a><a name="p189045212538"></a>srcShape1</p>
</td>
<td class="cellrowborder" valign="top" width="10.23%" headers="mcps1.2.4.1.2 "><p id="p490205216537"><a name="p490205216537"></a><a name="p490205216537"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.88%" headers="mcps1.2.4.1.3 "><p id="p290155216536"><a name="p290155216536"></a><a name="p290155216536"></a>输入srcTensor1的shape信息。</p>
</td>
</tr>
<tr id="row12913810416"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p1291148174119"><a name="p1291148174119"></a><a name="p1291148174119"></a>srcShape2</p>
</td>
<td class="cellrowborder" valign="top" width="10.23%" headers="mcps1.2.4.1.2 "><p id="p12911688418"><a name="p12911688418"></a><a name="p12911688418"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.88%" headers="mcps1.2.4.1.3 "><p id="p725218574416"><a name="p725218574416"></a><a name="p725218574416"></a>输入srcTensor2的shape信息。</p>
</td>
</tr>
<tr id="row1234411294113"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p1834491215418"><a name="p1834491215418"></a><a name="p1834491215418"></a>typeIsInt</p>
</td>
<td class="cellrowborder" valign="top" width="10.23%" headers="mcps1.2.4.1.2 "><p id="p173441912134115"><a name="p173441912134115"></a><a name="p173441912134115"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.88%" headers="mcps1.2.4.1.3 "><p id="p33449129417"><a name="p33449129417"></a><a name="p33449129417"></a>bool类型，true表示输入是int32_t。</p>
</td>
</tr>
<tr id="row4492162642313"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p690155215538"><a name="p690155215538"></a><a name="p690155215538"></a>typeSize</p>
</td>
<td class="cellrowborder" valign="top" width="10.23%" headers="mcps1.2.4.1.2 "><p id="p790195215536"><a name="p790195215536"></a><a name="p790195215536"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.88%" headers="mcps1.2.4.1.3 "><p id="p1492826172315"><a name="p1492826172315"></a><a name="p1492826172315"></a>输入的数据类型大小，单位为字节。比如输入的数据类型为half，此处应传入2。</p>
</td>
</tr>
<tr id="row8919193017270"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p4454173917272"><a name="p4454173917272"></a><a name="p4454173917272"></a>isReuseSource</p>
</td>
<td class="cellrowborder" valign="top" width="10.23%" headers="mcps1.2.4.1.2 "><p id="p19207307272"><a name="p19207307272"></a><a name="p19207307272"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.88%" headers="mcps1.2.4.1.3 "><p id="p159203303279"><a name="p159203303279"></a><a name="p159203303279"></a>是否复用源操作数输入的空间，与Power接口一致。</p>
</td>
</tr>
<tr id="row123907134016"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p119016528533"><a name="p119016528533"></a><a name="p119016528533"></a>maxValue</p>
</td>
<td class="cellrowborder" valign="top" width="10.23%" headers="mcps1.2.4.1.2 "><p id="p590155210531"><a name="p590155210531"></a><a name="p590155210531"></a>输出</p>
<p id="p13901052205318"><a name="p13901052205318"></a><a name="p13901052205318"></a></p>
</td>
<td class="cellrowborder" valign="top" width="72.88%" headers="mcps1.2.4.1.3 "><p id="p1890195216531"><a name="p1890195216531"></a><a name="p1890195216531"></a>Power接口能完成计算所需的最大临时空间大小，超出该值的空间不会被该接口使用。在最小临时空间-最大临时空间范围内，随着临时空间增大，kernel侧接口计算性能会有一定程度的优化提升。为了达到更好的性能，开发者可以根据实际的内存使用情况进行空间预留/申请。最大空间大小为0表示计算不需要临时空间。</p>
<p id="p7229175774011"><a name="p7229175774011"></a><a name="p7229175774011"></a>请注意，maxValue仅作为参考值，有可能大于<span id="ph1088254310583"><a name="ph1088254310583"></a><a name="ph1088254310583"></a>Unified Buffer</span>剩余空间的大小，该场景下，开发者需要根据<span id="ph20796650203718"><a name="ph20796650203718"></a><a name="ph20796650203718"></a>Unified Buffer</span>剩余空间的大小来选取合适的临时空间大小。</p>
</td>
</tr>
<tr id="row66944015289"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p1369580162813"><a name="p1369580162813"></a><a name="p1369580162813"></a>minValue</p>
</td>
<td class="cellrowborder" valign="top" width="10.23%" headers="mcps1.2.4.1.2 "><p id="p176953052810"><a name="p176953052810"></a><a name="p176953052810"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="72.88%" headers="mcps1.2.4.1.3 "><p id="p66955018287"><a name="p66955018287"></a><a name="p66955018287"></a>Power接口能完成计算所需最小临时空间大小。为保证功能正确，接口计算时预留/申请的临时空间不能小于该数值。最小空间大小为0表示计算不需要临时空间。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="zh-cn_topic_0000001867289929_zh-cn_topic_0000001389787297_section19165124931511"></a>

无

## 调用示例<a name="section642mcpsimp"></a>

完整的调用样例请参考[更多样例](更多样例-104.md)。

-   Power\(dstTensor, srcTensor1, srcTensor2\)样例

    ```
    // srcTensor1、srcTensor2输入shape信息均为512;算子输入的数据类型为float;不允许修改源操作数
    std::vector<int64_t> shape_vec = {1024};
    ge::Shape shape(shape_vec);
    uint32_t maxValue = 0;
    uint32_t minValue = 0;
    AscendC::GetPowerMaxMinTmpSize(shape, shape, false, 4, false, maxValue, minValue);
    ```

-   Power\(dstTensor, srcTensor1, scalarValue\)样例

    ```
    // srcTensor1输入shape信息为128*128，scalarValue的shape为1;算子输入的数据类型为half;不允许修改源操作数
    std::vector<int64_t> shape1_vec = {128,128};
    std::vector<int64_t> shape2_vec = {1};
    ge::Shape shape1(shape1_vec);
    ge::Shape shape2(shape2_vec);
    uint32_t maxValue = 0;
    uint32_t minValue = 0;
    AscendC::GetPowerMaxMinTmpSize(shape1, shape2, false, 2, false, maxValue, minValue);
    ```

-   Power\(dstTensor, scalarValue, srcTensor2\)样例

    ```
    //scalarValue的shape为1，srcTensor2输入shape信息为128*128;算子输入的数据类型为float;不允许修改源操作数
    std::vector<int64_t> shape1_vec = {1};
    std::vector<int64_t> shape2_vec = {128,128};
    ge::Shape shape1(shape1_vec);
    ge::Shape shape2(shape2_vec);
    uint32_t maxValue = 0;
    uint32_t minValue = 0;
    AscendC::GetPowerMaxMinTmpSize(shape1, shape2, false, 4, false, maxValue, minValue);
    ```

