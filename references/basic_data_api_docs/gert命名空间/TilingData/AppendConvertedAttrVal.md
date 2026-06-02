# AppendConvertedAttrVal

**页面ID:** atlasopapi_07_00262  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00262.html

---

#### 函数功能

将RuntimeAttrs内存中的第index个属性，从src_type数据类型转成dst_type数据类型后，追加在当前的tilingdata后面。

#### 函数原型

**ge::graphStatus AppendConvertedAttrVal(const RuntimeAttrs *attrs, const size_t attr_index, const AttrDataType src_type, const AttrDataType dst_type)**

#### 参数说明

**表1 **参数列表

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| attrs | 输入 | 图执行时的属性。 |
| attr_index | 输入 | 属性的序号。 |
| src_type | 输入 | 属性的原始dtype。AttrDataType类型，具体定义如下： ``` enum class AttrDataType : int32_t {   kBool = 0,   kString,   kInt32,   kInt64,   kUint32,   kFloat32,   kFloat16,   kListBool,   kListString,   kListInt32,   kListInt64,   kListUint32,   kListFloat32,   kListFloat16,   kListListInt32,   kListListInt64,   kBfloat16,   kInt8,   kInt16,   kListInt8,   kListInt16,   kTypeEnd }; ``` src_type和dst_type支持的取值范围和对应关系见表2。 |
| dst_type | 输入 | 属性的目标dtype。 src_type和dst_type支持的取值范围和对应关系见表2。 |

**表2 **src_type和dst_type支持的取值范围和对应关系

| src_type | dst_type |
| --- | --- |
| kInt64 | kInt32、kUint32 |
| kListInt64 | kListInt32、kListUint32 |
| kInt32 | kInt32 |
| kListInt32 | kListInt32 |
| kFloat32 | kFloat32、kFloat16、kBfloat16、kInt64、kInt32、kInt16、kInt8 |
| kListFloat32 | kListFloat32、kListFloat16、kListInt64、kListInt32、kListInt16、kListInt8 |
| kBool | kBool |
| kString | kString |

#### 返回值说明

- 成功返回：ge::GRAPH_SUCCESS。
- 失败返回：ge::GRAPH_FAILED。

#### 约束说明

无。

#### 调用示例

```
auto td_buf = TilingData::CreateCap(100U);
auto td = reinterpret_cast<TilingData *>(td_buf.get());

auto holder = BuildTestContext();
auto context = holder.GetContext<TilingContext>();
auto ret = td->AppendConvertedAttrVal(context->GetAttrs(), 1, AttrDataType::kString, AttrDataType::kString)

FakeKernelContextHolder BuildTestContext() {
  auto holder = gert::KernelRunContextFaker()
                    .NodeIoNum(1, 1)
                    .IrInputNum(1)
                    .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
                    .NodeAttrs({{"int", ge::AnyValue::CreateFrom<int64_t>(0x7fffffffUL)},
                                {"str", ge::AnyValue::CreateFrom<std::string>("Hello!")},
                                {"bool", ge::AnyValue::CreateFrom<bool>(true)},
                                {"float", ge::AnyValue::CreateFrom<float>(10.101)},
                                {"list_int", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2, 3})},
                                {"list_int2", ge::AnyValue::CreateFrom<std::vector<int64_t>>({4, 5, 6})},
                                {"list_float", ge::AnyValue::CreateFrom<std::vector<float>>({1.2, 3.4, 4.5})}})
                    .Build();
  return holder;
}
```
