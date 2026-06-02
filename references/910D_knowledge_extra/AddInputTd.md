# AddInputTd<a name="ZH-CN_TOPIC_0000002554343941"></a>

## 功能说明<a name="zh-cn_topic_0000001867409737_zh-cn_topic_0000001389787297_section36583473819"></a>

为算子增加输入Tensor的描述

## 函数原型<a name="zh-cn_topic_0000001867409737_zh-cn_topic_0000001389787297_section13230182415108"></a>

```
ContextBuilder &AddInputTd(int32_t index, ge::DataType dtype, ge::Format originFormat,
ge::Format storageFormat, gert::StorageShape storageShape)
ContextBuilder &AddInputTd(int32_t index, ge::DataType dtype, ge::Format originFormat,
ge::Format storageFormat, gert::StorageShape storageShape, void* constValues)
ContextBuilder &AddInputTd(int32_t index, ge::DataType dtype, ge::Format originFormat,
ge::Format storageFormat, gert::StorageShape storageShape, const std::string &filePath)
```

## 参数说明<a name="zh-cn_topic_0000001867409737_zh-cn_topic_0000001389787297_section75395119104"></a>

<a name="zh-cn_topic_0000001867409737_zh-cn_topic_0000001389787297_table111938719446"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001867409737_zh-cn_topic_0000001389787297_row6223476444"><th class="cellrowborder" valign="top" width="17.22%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001867409737_zh-cn_topic_0000001389787297_p10223674448"><a name="zh-cn_topic_0000001867409737_zh-cn_topic_0000001389787297_p10223674448"></a><a name="zh-cn_topic_0000001867409737_zh-cn_topic_0000001389787297_p10223674448"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="15.340000000000002%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001867409737_zh-cn_topic_0000001389787297_p645511218169"><a name="zh-cn_topic_0000001867409737_zh-cn_topic_0000001389787297_p645511218169"></a><a name="zh-cn_topic_0000001867409737_zh-cn_topic_0000001389787297_p645511218169"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="67.44%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001867409737_zh-cn_topic_0000001389787297_p1922337124411"><a name="zh-cn_topic_0000001867409737_zh-cn_topic_0000001389787297_p1922337124411"></a><a name="zh-cn_topic_0000001867409737_zh-cn_topic_0000001389787297_p1922337124411"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001867409737_zh-cn_topic_0000001389787297_row152234713443"><td class="cellrowborder" valign="top" width="17.22%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001867409737_zh-cn_topic_0000001389787297_p8563195616313"><a name="zh-cn_topic_0000001867409737_zh-cn_topic_0000001389787297_p8563195616313"></a><a name="zh-cn_topic_0000001867409737_zh-cn_topic_0000001389787297_p8563195616313"></a>index</p>
</td>
<td class="cellrowborder" valign="top" width="15.340000000000002%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001867409737_zh-cn_topic_0000001389787297_p15663137127"><a name="zh-cn_topic_0000001867409737_zh-cn_topic_0000001389787297_p15663137127"></a><a name="zh-cn_topic_0000001867409737_zh-cn_topic_0000001389787297_p15663137127"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.44%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001867409737_zh-cn_topic_0000001389787297_p2684123934216"><a name="zh-cn_topic_0000001867409737_zh-cn_topic_0000001389787297_p2684123934216"></a><a name="zh-cn_topic_0000001867409737_zh-cn_topic_0000001389787297_p2684123934216"></a>算子输入索引，从0开始计数。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001867409737_row19461822171616"><td class="cellrowborder" valign="top" width="17.22%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001867409737_p1346122271616"><a name="zh-cn_topic_0000001867409737_p1346122271616"></a><a name="zh-cn_topic_0000001867409737_p1346122271616"></a>dtype</p>
</td>
<td class="cellrowborder" valign="top" width="15.340000000000002%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001867409737_p1746122181614"><a name="zh-cn_topic_0000001867409737_p1746122181614"></a><a name="zh-cn_topic_0000001867409737_p1746122181614"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.44%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001867409737_p1246122131616"><a name="zh-cn_topic_0000001867409737_p1246122131616"></a><a name="zh-cn_topic_0000001867409737_p1246122131616"></a>算子输入tensor的数据类型</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001867409737_row969015712610"><td class="cellrowborder" valign="top" width="17.22%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001867409737_p156905712261"><a name="zh-cn_topic_0000001867409737_p156905712261"></a><a name="zh-cn_topic_0000001867409737_p156905712261"></a>originFormat</p>
</td>
<td class="cellrowborder" valign="top" width="15.340000000000002%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001867409737_p46905752616"><a name="zh-cn_topic_0000001867409737_p46905752616"></a><a name="zh-cn_topic_0000001867409737_p46905752616"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.44%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001867409737_p1469017713265"><a name="zh-cn_topic_0000001867409737_p1469017713265"></a><a name="zh-cn_topic_0000001867409737_p1469017713265"></a>算子输入tensor原始格式</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001867409737_row3613913838"><td class="cellrowborder" valign="top" width="17.22%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001867409737_p1161416131635"><a name="zh-cn_topic_0000001867409737_p1161416131635"></a><a name="zh-cn_topic_0000001867409737_p1161416131635"></a>storageFormat</p>
</td>
<td class="cellrowborder" valign="top" width="15.340000000000002%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001867409737_p1561419131733"><a name="zh-cn_topic_0000001867409737_p1561419131733"></a><a name="zh-cn_topic_0000001867409737_p1561419131733"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.44%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001867409737_p76146131930"><a name="zh-cn_topic_0000001867409737_p76146131930"></a><a name="zh-cn_topic_0000001867409737_p76146131930"></a>算子输入tensor运行时格式</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001867409737_row1158814326319"><td class="cellrowborder" valign="top" width="17.22%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001867409737_p195897321830"><a name="zh-cn_topic_0000001867409737_p195897321830"></a><a name="zh-cn_topic_0000001867409737_p195897321830"></a>storageShape</p>
</td>
<td class="cellrowborder" valign="top" width="15.340000000000002%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001867409737_p115897323312"><a name="zh-cn_topic_0000001867409737_p115897323312"></a><a name="zh-cn_topic_0000001867409737_p115897323312"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.44%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001867409737_p858912321435"><a name="zh-cn_topic_0000001867409737_p858912321435"></a><a name="zh-cn_topic_0000001867409737_p858912321435"></a>算子输入tensor的shape</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001867409737_row8480150635"><td class="cellrowborder" valign="top" width="17.22%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001867409737_p15480115017315"><a name="zh-cn_topic_0000001867409737_p15480115017315"></a><a name="zh-cn_topic_0000001867409737_p15480115017315"></a>constValues</p>
</td>
<td class="cellrowborder" valign="top" width="15.340000000000002%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001867409737_p548019501938"><a name="zh-cn_topic_0000001867409737_p548019501938"></a><a name="zh-cn_topic_0000001867409737_p548019501938"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.44%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001867409737_p3480185014314"><a name="zh-cn_topic_0000001867409737_p3480185014314"></a><a name="zh-cn_topic_0000001867409737_p3480185014314"></a>值依赖场景下该输入tensor需要设置的数据指针。 bfloat16与float16的数据依赖场景，请传入float格式的数据，接口内部自行转换成bfloat16或float16。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001867409737_row1385811187"><td class="cellrowborder" valign="top" width="17.22%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001867409737_p19386181387"><a name="zh-cn_topic_0000001867409737_p19386181387"></a><a name="zh-cn_topic_0000001867409737_p19386181387"></a>filePath</p>
</td>
<td class="cellrowborder" valign="top" width="15.340000000000002%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001867409737_p438615118815"><a name="zh-cn_topic_0000001867409737_p438615118815"></a><a name="zh-cn_topic_0000001867409737_p438615118815"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.44%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001867409737_p1138610117817"><a name="zh-cn_topic_0000001867409737_p1138610117817"></a><a name="zh-cn_topic_0000001867409737_p1138610117817"></a>值依赖场景下该输入tensor的bin格式数据文件路径</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0000001867409737_zh-cn_topic_0000001389787297_section25791320141317"></a>

当前ContextBuilder的对象。

## 约束说明<a name="zh-cn_topic_0000001867409737_zh-cn_topic_0000001389787297_section19165124931511"></a>

输入的index需要基于算子IR定义，按照IrInstanceNum声明顺序来排布；

调用AddInputTd前需要调用NodeIoNum与IrInstanceNum接口

## 调用示例<a name="zh-cn_topic_0000001867409737_zh-cn_topic_0000001389787297_section320753512363"></a>

```
gert::StorageShape x_shape = {{1024, 5120}, {1024, 5120}};
gert::StorageShape expert_tokens_shape = {{16}, {16}};
gert::StorageShape weight1_shape = {{16, 5120, 0}, {16, 5120, 0}};
gert::StorageShape bias1_shape = {{16, 0}, {16, 0}};

std::vector<float> x_const_value (1024 * 5120, 2.f);
std::vector<float> bias_value (16 * 5120, 3.f);
context_ascendc::ContextBuilder builder
(void)builder.NodeIoNum(5, 1) // 声明算子有5个输入，1个输出
    .IrInstanceNum({1, 1, 2, 1, 1}) // 声明index 2的算子tensor有两个dynamic实例
    .SetOpNameType("tmpName", "tmpType")
    .AddInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND, x_shape, reinterpret_cast<void *>(x_const_value.data()))  // 内部会将该指针指向的数据转为float16类型
    .AddInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND, weight1_shape)
    .AddInputTd(2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND, expert_tokens_shape, "./expert_tokens_data.bin")   // index2 的 第一个dynamic tensor，值依赖场景传入数据路径
    .AddInputTd(3, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND, expert_tokens_shape, "./expert_tokens_data.bin")   // index2 的 第二个dynamic tensor， 值依赖场景传入数据路径
    .AddInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND, bias1_shape)
    .AddInputTd(5, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND, bias2_shape, reinterpret_cast<void*>(bias_value.data()))  // 内部会将该指针指向的数据转为Bf16类型

```

