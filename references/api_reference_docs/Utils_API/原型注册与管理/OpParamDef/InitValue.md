# InitValue

**页面ID:** atlasascendc_api_07_0972  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0972.html

---

#### 功能说明

通过该host侧接口设置算子输出的初始值，设置后会在算子执行前对算子输出的GM空间进行清零操作或者插入memset类算子进行初始值设置。

**InitValue和**SetNeedAtomic**接口配合使用，SetNeedAtomic接口需要配置为true。**

#### 函数原型

- 在算子执行前，对输出参数对应GM空间进行清零。

```
OpParamDef &InitValue(uint64_t value)
```

- 指定输出参数初值的类型和值，输出参数调用该接口，会在算子执行前，对输出参数对应GM空间插入对应类型和值的memset类算子。

```
OpParamDef &InitValue(const ScalarVar &value)
```

- 指定输出参数初值类型和值的列表，依次对应输出参数的数据类型和数据格式组合，会在算子执行前，对输出参数对应GM空间插入对应类型和值的memset类算子。

```
OpParamDef &InitValue(const std::vector<ScalarVar> &value)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| - uint64_t类型参数            仅支持输入0，输出参数调用该接口，会在算子执行前，对输出参数对应GM空间进行清零。           - ScalarVar类型参数            ScalarVar用于指定输出参数初值的类型ScalarType和值ScalarNum，具体定义如下：                                                                                                                                             ``` enum class ScalarType : uint32_t {   UINT64 = 0,   INT64 = 1,   UINT32 = 2,   INT32 = 3,   UINT16 = 4,   INT16 = 5,   UINT8 = 6,   INT8 = 7,   FLOAT32 = 8,   FLOAT16 = 9,   INVALID_DTYPE = static_cast<uint32_t>(-1), }; union ScalarNum {   uint64_t value_u64;   int64_t value_i64;   float value_f32;   ScalarNum() : value_u64(0) {}   explicit ScalarNum(uint64_t value) : value_u64(value) {}   explicit ScalarNum(int64_t value) : value_i64(value) {}   explicit ScalarNum(float value) : value_f32(value) {} }; struct ScalarVar {   ScalarType scalar_type;   ScalarNum scalar_num;   ScalarVar();   ScalarVar(ScalarType type, uint64_t num);   ScalarVar(ScalarType type, int64_t num);   ScalarVar(ScalarType type, int num);   ScalarVar(ScalarType type, unsigned int num);   ScalarVar(ScalarType type, float num);   ScalarVar(ScalarType type, double num);   bool operator==(const ScalarVar& other) const; }; ```                                                                                                    ScalarType当前仅支持UINT64/INT64/UINT32/INT32/UINT16/INT16/UINT8/INT8/FLOAT32/FLOAT16;            ScalarNum支持uint64_t/int64_t/float类型。            为方便使用，ScalarVar也支持立即数初始化，示例如下：                                                                                                                                             ``` InitValue({ScalarType::INT16, 1}); ```                                                                                                   - const std::vector<ScalarVar> &value类型            指定输出参数初值类型和值的列表，依次对应输出参数的数据类型和数据格式组合。 |  |  |

#### 返回值说明

OpParamDef算子定义，OpParamDef请参考OpParamDef。

#### 约束说明

- InitValue和SetNeedAtomic接口配合使用，否则会出现初始化不生效的情况。
- 针对如下产品型号：

         Atlas A3 训练系列产品
        /
         Atlas A3 推理系列产品

         Atlas A2 训练系列产品
        /
         Atlas A2 推理系列产品

         Atlas 200I/500 A2 推理产品

         Atlas 推理系列产品

         Atlas 训练系列产品

插入memset类算子时，仅在入图场景下支持初始值设置任意值，单算子API执行的场景下仅支持清零。

- 针对OpParamDef &InitValue(uint64_t value)接口，算子输出参数的数据类型支持范围如下：UINT64/INT64/UINT32/INT32/UINT16/INT16/UINT8/INT8/FLOAT32/FLOAT16，超出该范围为未定义行为。
- 针对OpParamDef &InitValue(const std::vector<ScalarVar> &value)接口输入value的size需要与输出参数配置的DataType或DataTypeList接口参数的size一致。同时，相同数据类型需保证设置的类型和值相同，否则将会报错。
- 对于同一个输出参数仅支持调用一种接口设置初值，调用多种InitValue接口为未定义行为；多次调用同一种接口以最后一次调用设置的初值为准。
- 基于旧版本CANN包（不支持InitValue特性）生成的自定义算子工程，无法兼容InitValue接口。在使用非当前版本CANN包生成的自定义算子工程时，需特别注意兼容性问题。您可以通过查看自定义算子工程下cmake/util/ascendc_impl_build.py中有无output_init_value字段来确认当前工程是否支持该特性，如果未找到该字段，则需要重新生成自定义算子工程以启用InitValue特性。

#### 调用示例

```
// OpParamDef &InitValue(uint64_t value)示例
this->Output("z")
     .ParamType(REQUIRED)
     .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32})
     .FormatList({ ge::FORMAT_ND})
     .InitValue(0);

// OpParamDef &InitValue(const ScalarVar &value)示例
this->Output("z")
     .ParamType(REQUIRED)
     .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32})
     .FormatList({ ge::FORMAT_ND})
     .InitValue({ScalarType::INT16, 1});

// OpParamDef &InitValue(const std::vector<ScalarVar> &value)示例
this->Output("z")
     .ParamType(REQUIRED)
     .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32})
     .FormatList({ ge::FORMAT_ND})
     .InitValue({{ScalarType::INT16, 1}, {ScalarType::FLOAT32, 3.2}, {ScalarType::INT64, 7}});

this->Output("z")
     .ParamType(REQUIRED)
     .DataType({ge::DT_INT32, ge::DT_FLOAT, ge::DT_INT32})  // 第一个和第三个DataType相同
     .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_NHWC})
     .InitValue({{ScalarType::INT16, 1}, {ScalarType::FLOAT32, 3.2}, {ScalarType::INT16, 1}}); // InitValue对应的数据类型和数值也需相同
```
