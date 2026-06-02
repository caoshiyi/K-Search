# DataType

**页面ID:** atlasopapi_07_00514  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00514.html

---

DataType枚举值定义如下：

```
enum DataType {
  DT_FLOAT = 0,            // float type
  DT_FLOAT16 = 1,          // fp16 type
  DT_INT8 = 2,             // int8 type
  DT_INT32 = 3,            // int32 type
  DT_UINT8 = 4,            // uint8 type
                           // reserved
  DT_INT16 = 6,            // int16 type
  DT_UINT16 = 7,           // uint16 type
  DT_UINT32 = 8,           // unsigned int32
  DT_INT64 = 9,            // int64 type
  DT_UINT64 = 10,          // unsigned int64
  DT_DOUBLE = 11,          // double type
  DT_BOOL = 12,            // bool type
  DT_STRING = 13,          // string type
  DT_DUAL_SUB_INT8 = 14,   // dual output int8 type
  DT_DUAL_SUB_UINT8 = 15,  // dual output uint8 type
  DT_COMPLEX64 = 16,       // complex64 type
  DT_COMPLEX128 = 17,      // complex128 type
  DT_QINT8 = 18,           // qint8 type
  DT_QINT16 = 19,          // qint16 type
  DT_QINT32 = 20,          // qint32 type
  DT_QUINT8 = 21,          // quint8 type
  DT_QUINT16 = 22,         // quint16 type
  DT_RESOURCE = 23,        // resource type
  DT_STRING_REF = 24,      // string ref type
  DT_DUAL = 25,            // dual output type
  DT_VARIANT = 26,         // dt_variant type
  DT_BF16 = 27,            // bf16 type
  DT_UNDEFINED = 28,       // Used to indicate a DataType field has not been set.
  DT_INT4 = 29,            // int4 type
  DT_UINT1 = 30,           // uint1 type
  DT_INT2 = 31,            // int2 type
  DT_UINT2 = 32,           // uint2 type
  DT_COMPLEX32 = 33,       // complex32 type
  DT_HIFLOAT8 = 34,        // hifloat8 type       当前版本不支持该类型。
  DT_FLOAT8_E5M2 = 35,     // float8_e5m2 type     当前版本不支持该类型。
  DT_FLOAT8_E4M3FN = 36,   // float8_e4m3fn type   当前版本不支持该类型。
  DT_FLOAT8_E8M0 = 37,     // float8_e8m0 type     当前版本不支持该类型。
  DT_FLOAT6_E3M2 = 38,     // float6_e3m2 type     当前版本不支持该类型。
  DT_FLOAT6_E2M3 = 39,     // float6_e2m3 type     当前版本不支持该类型。
  DT_FLOAT4_E2M1 = 40,     // float4_e2m1 type     当前版本不支持该类型。
  DT_FLOAT4_E1M2 = 41,     // float4_e1m2 type     当前版本不支持该类型。
  DT_MAX                   // Mark the boundaries of data types
};
```
