# C_DataType

**页面ID:** atlasopapi_07_00719  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00719.html

---

C_DataType枚举值定义如下：

```
typedef enum {
  C_DT_FLOAT = 0,            // float type
  C_DT_FLOAT16 = 1,          // fp16 type
  C_DT_INT8 = 2,             // int8 type
  C_DT_INT32 = 3,            // int32 type
  C_DT_UINT8 = 4,            // uint8 type
  // reserved
  C_DT_INT16 = 6,            // int16 type
  C_DT_UINT16 = 7,           // uint16 type
  C_DT_UINT32 = 8,           // unsigned int32
  C_DT_INT64 = 9,            // int64 type
  C_DT_UINT64 = 10,          // unsigned int64
  C_DT_DOUBLE = 11,          // double type
  C_DT_BOOL = 12,            // bool type
  C_DT_STRING = 13,          // string type
  C_DT_DUAL_SUB_INT8 = 14,   // dual output int8 type
  C_DT_DUAL_SUB_UINT8 = 15,  // dual output uint8 type
  C_DT_COMPLEX64 = 16,       // complex64 type
  C_DT_COMPLEX128 = 17,      // complex128 type
  C_DT_QINT8 = 18,           // qint8 type
  C_DT_QINT16 = 19,          // qint16 type
  C_DT_QINT32 = 20,          // qint32 type
  C_DT_QUINT8 = 21,          // quint8 type
  C_DT_QUINT16 = 22,         // quint16 type
  C_DT_RESOURCE = 23,        // resource type
  C_DT_STRING_REF = 24,      // string ref type
  C_DT_DUAL = 25,            // dual output type
  C_DT_VARIANT = 26,         // dt_variant type
  C_DT_BF16 = 27,            // bf16 type
  C_DT_UNDEFINED = 28,       // Used to indicate a DataType field has not been set.
  C_DT_INT4 = 29,            // int4 type
  C_DT_UINT1 = 30,           // uint1 type
  C_DT_INT2 = 31,            // int2 type
  C_DT_UINT2 = 32,           // uint2 type
  C_DT_COMPLEX32 = 33,       // complex32 type
  C_DT_HIFLOAT8 = 34,        // hifloat8 type     当前版本不支持该类型。
  C_DT_FLOAT8_E5M2 = 35,     // float8_e5m2 type  当前版本不支持该类型。
  C_DT_FLOAT8_E4M3FN = 36,   // float8_e4m3fn type当前版本不支持该类型。
  C_DT_FLOAT8_E8M0 = 37,     // float8_e8m0 type   当前版本不支持该类型。
  C_DT_FLOAT6_E3M2 = 38,     // float6_e3m2 type   当前版本不支持该类型。
  C_DT_FLOAT6_E2M3 = 39,     // float6_e2m3 type    当前版本不支持该类型。
  C_DT_FLOAT4_E2M1 = 40,     // float4_e2m1 type    当前版本不支持该类型。
  C_DT_FLOAT4_E1M2 = 41,     // float4_e1m2 type    当前版本不支持该类型。
  C_DT_MAX                   // Mark the boundaries of data types
} C_DataType;
```
