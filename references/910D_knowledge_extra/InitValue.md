# InitValue<a name="ZH-CN_TOPIC_0000002523304802"></a>

## 功能说明<a name="zh-cn_topic_0000001960169216_zh-cn_topic_0000001526594958_zh-cn_topic_0000001525424352_section36583473819"></a>

通过该host侧接口设置算子输出的初始值，设置后会在算子执行前对算子输出的GM空间进行清零操作或者插入memset类算子进行初始值设置。

**InitValue和SetNeedAtomic接口配合使用，SetNeedAtomic接口需要配置为true。**

## 函数原型<a name="zh-cn_topic_0000001960169216_zh-cn_topic_0000001526594958_zh-cn_topic_0000001525424352_section13230182415108"></a>

-   在算子执行前，对输出参数对应GM空间进行清零。

    ```
    OpParamDef &InitValue(uint64_t value)
    ```

-   指定输出参数初值的类型和值，输出参数调用该接口，会在算子执行前，对输出参数对应GM空间插入对应类型和值的memset类算子。

    ```
    OpParamDef &InitValue(const ScalarVar &value)
    ```

-   指定输出参数初值类型和值的列表，依次对应输出参数的数据类型和数据格式组合，会在算子执行前，对输出参数对应GM空间插入对应类型和值的memset类算子。

    ```
    OpParamDef &InitValue(const std::vector<ScalarVar> &value)
    ```

## 参数说明<a name="zh-cn_topic_0000001960169216_zh-cn_topic_0000001526594958_zh-cn_topic_0000001525424352_section75395119104"></a>

<a name="zh-cn_topic_0000001960169216_zh-cn_topic_0000001526594958_zh-cn_topic_0000001575944081_table111938719446"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001960169216_zh-cn_topic_0000001526594958_zh-cn_topic_0000001575944081_row6223476444"><th class="cellrowborder" valign="top" width="17.22%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001960169216_zh-cn_topic_0000001526594958_zh-cn_topic_0000001575944081_p10223674448"><a name="zh-cn_topic_0000001960169216_zh-cn_topic_0000001526594958_zh-cn_topic_0000001575944081_p10223674448"></a><a name="zh-cn_topic_0000001960169216_zh-cn_topic_0000001526594958_zh-cn_topic_0000001575944081_p10223674448"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="15.340000000000002%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001960169216_zh-cn_topic_0000001526594958_zh-cn_topic_0000001575944081_p645511218169"><a name="zh-cn_topic_0000001960169216_zh-cn_topic_0000001526594958_zh-cn_topic_0000001575944081_p645511218169"></a><a name="zh-cn_topic_0000001960169216_zh-cn_topic_0000001526594958_zh-cn_topic_0000001575944081_p645511218169"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="67.44%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001960169216_zh-cn_topic_0000001526594958_zh-cn_topic_0000001575944081_p1922337124411"><a name="zh-cn_topic_0000001960169216_zh-cn_topic_0000001526594958_zh-cn_topic_0000001575944081_p1922337124411"></a><a name="zh-cn_topic_0000001960169216_zh-cn_topic_0000001526594958_zh-cn_topic_0000001575944081_p1922337124411"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001960169216_zh-cn_topic_0000001526594958_zh-cn_topic_0000001575944081_row152234713443"><td class="cellrowborder" valign="top" width="17.22%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001960169216_p466181582315"><a name="zh-cn_topic_0000001960169216_p466181582315"></a><a name="zh-cn_topic_0000001960169216_p466181582315"></a>value</p>
</td>
<td class="cellrowborder" valign="top" width="15.340000000000002%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001960169216_p1464218131118"><a name="zh-cn_topic_0000001960169216_p1464218131118"></a><a name="zh-cn_topic_0000001960169216_p1464218131118"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.44%" headers="mcps1.1.4.1.3 "><a name="zh-cn_topic_0000001960169216_ul130045510411"></a><a name="zh-cn_topic_0000001960169216_ul130045510411"></a><ul id="zh-cn_topic_0000001960169216_ul130045510411"><li>uint64_t类型参数<p id="zh-cn_topic_0000001960169216_p1920504112421"><a name="zh-cn_topic_0000001960169216_p1920504112421"></a><a name="zh-cn_topic_0000001960169216_p1920504112421"></a>仅支持输入0，输出参数调用该接口，会在算子执行前，对输出参数对应GM空间进行清零。</p>
</li><li>ScalarVar类型参数<p id="zh-cn_topic_0000001960169216_p328318251747"><a name="zh-cn_topic_0000001960169216_p328318251747"></a><a name="zh-cn_topic_0000001960169216_p328318251747"></a>ScalarVar用于指定输出参数初值的类型ScalarType和值ScalarNum，具体定义如下：</p>
<a name="zh-cn_topic_0000001960169216_screen1644815588715"></a><a name="zh-cn_topic_0000001960169216_screen1644815588715"></a><pre class="screen" codetype="Cpp" id="zh-cn_topic_0000001960169216_screen1644815588715">enum class ScalarType : uint32_t {
  UINT64 = 0,
  INT64 = 1,
  UINT32 = 2,
  INT32 = 3,
  UINT16 = 4,
  INT16 = 5,
  UINT8 = 6,
  INT8 = 7,
  FLOAT32 = 8,
  FLOAT16 = 9,
  INVALID_DTYPE = static_cast&lt;uint32_t&gt;(-1),
};
union ScalarNum {
  uint64_t value_u64;
  int64_t value_i64;
  float value_f32;
  ScalarNum() : value_u64(0) {}
  explicit ScalarNum(uint64_t value) : value_u64(value) {}
  explicit ScalarNum(int64_t value) : value_i64(value) {}
  explicit ScalarNum(float value) : value_f32(value) {}
};
struct ScalarVar {
  ScalarType scalar_type;
  ScalarNum scalar_num;
  ScalarVar();
  ScalarVar(ScalarType type, uint64_t num);
  ScalarVar(ScalarType type, int64_t num);
  ScalarVar(ScalarType type, int num);
  ScalarVar(ScalarType type, unsigned int num);
  ScalarVar(ScalarType type, float num);
  ScalarVar(ScalarType type, double num);
  bool operator==(const ScalarVar&amp; other) const;
};</pre>
<p id="zh-cn_topic_0000001960169216_p7872325173718"><a name="zh-cn_topic_0000001960169216_p7872325173718"></a><a name="zh-cn_topic_0000001960169216_p7872325173718"></a>ScalarType当前仅支持UINT64/INT64/UINT32/INT32/UINT16/INT16/UINT8/INT8/FLOAT32/FLOAT16;</p>
<p id="zh-cn_topic_0000001960169216_p17482719222"><a name="zh-cn_topic_0000001960169216_p17482719222"></a><a name="zh-cn_topic_0000001960169216_p17482719222"></a>ScalarNum支持uint64_t/int64_t/float类型。</p>
<p id="zh-cn_topic_0000001960169216_p14507521395"><a name="zh-cn_topic_0000001960169216_p14507521395"></a><a name="zh-cn_topic_0000001960169216_p14507521395"></a>为方便使用，ScalarVar也支持立即数初始化，示例如下：</p>
<a name="zh-cn_topic_0000001960169216_screen9280101782212"></a><a name="zh-cn_topic_0000001960169216_screen9280101782212"></a><pre class="screen" codetype="Cpp" id="zh-cn_topic_0000001960169216_screen9280101782212">InitValue({ScalarType::INT16, 1});</pre>
</li><li>const std::vector&lt;ScalarVar&gt; &amp;value类型<p id="zh-cn_topic_0000001960169216_p7183525183613"><a name="zh-cn_topic_0000001960169216_p7183525183613"></a><a name="zh-cn_topic_0000001960169216_p7183525183613"></a>指定输出参数初值类型和值的列表，依次对应输出参数的数据类型和数据格式组合。</p>
</li></ul>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0000001960169216_zh-cn_topic_0000001526594958_zh-cn_topic_0000001525424352_section25791320141317"></a>

OpParamDef算子定义，OpParamDef请参考[OpParamDef](OpParamDef.md)。

## 约束说明<a name="zh-cn_topic_0000001960169216_zh-cn_topic_0000001526594958_zh-cn_topic_0000001525424352_section19165124931511"></a>

-   InitValue和SetNeedAtomic接口配合使用，否则会出现初始化不生效的情况。
-   针对OpParamDef &InitValue\(uint64\_t value\)接口，算子输出参数的数据类型支持范围如下：UINT64/INT64/UINT32/INT32/UINT16/INT16/UINT8/INT8/FLOAT32/FLOAT16，超出该范围为未定义行为。
-   针对OpParamDef &InitValue\(const std::vector<ScalarVar\> &value\)接口输入value的size需要与输出参数配置的[DataType](DataType.md)或[DataTypeList](DataTypeList.md)接口参数的size一致。同时，相同数据类型需保证设置的类型和值相同，否则将会报错。
-   对于同一个输出参数仅支持调用一种接口设置初值，调用多种InitValue接口为未定义行为；多次调用同一种接口以最后一次调用设置的初值为准。
-   基于旧版本CANN包（不支持InitValue特性）生成的自定义算子工程，无法兼容InitValue接口。在使用非当前版本CANN包生成的自定义算子工程时，需特别注意兼容性问题。您可以通过查看自定义算子工程下cmake/util/ascendc\_impl\_build.py中有无output\_init\_value字段来确认当前工程是否支持该特性，如果未找到该字段，则需要重新生成自定义算子工程以启用InitValue特性。

## 调用示例<a name="zh-cn_topic_0000001960169216_zh-cn_topic_0000001526594958_zh-cn_topic_0000001575944081_section320753512363"></a>

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

