# TensorType

**页面ID:** atlasopapi_07_00457  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00457.html

---

TensorType类用以定义输入或者输出支持的数据类型，TensorType提供以下接口指定支持的数据类型：

```
struct TensorType {
  explicit TensorType(DataType dt);

  TensorType(const std::initializer_list<DataType> &initial_types);

  static TensorType ALL() {
    return TensorType{DT_BOOL,   DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT,  DT_FLOAT16, DT_INT16,
                      DT_INT32,  DT_INT64,      DT_INT8,      DT_QINT16, DT_QINT32, DT_QINT8,   DT_QUINT16,
                      DT_QUINT8, DT_RESOURCE,   DT_STRING,    DT_UINT16, DT_UINT32, DT_UINT64,  DT_UINT8,
                      DT_BF16, DT_COMPLEX32};
  }

  static TensorType QuantifiedType() { return TensorType{DT_QINT16, DT_QINT32, DT_QINT8, DT_QUINT16, DT_QUINT8}; }

  static TensorType OrdinaryType() {
    return TensorType{DT_BOOL,  DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT,  DT_FLOAT16, DT_INT16,
                      DT_INT32, DT_INT64,      DT_INT8,      DT_UINT16, DT_UINT32, DT_UINT64,  DT_UINT8,
                      DT_BF16, DT_COMPLEX32};
  }

  static TensorType BasicType() {
    return TensorType{DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT,  DT_FLOAT16, DT_INT16,
                      DT_INT32,      DT_INT64,     DT_INT8,   DT_QINT16, DT_QINT32,  DT_QINT8,
                      DT_QUINT16,    DT_QUINT8,    DT_UINT16, DT_UINT32, DT_UINT64,  DT_UINT8,
                      DT_BF16, DT_COMPLEX32};
  }

  static TensorType NumberType() {
    return TensorType{DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT,  DT_FLOAT16, DT_INT16,  DT_INT32,  DT_INT64,
                      DT_INT8,       DT_QINT32,    DT_QINT8,  DT_QUINT8, DT_UINT16,  DT_UINT32, DT_UINT64, DT_UINT8,
                      DT_BF16, DT_COMPLEX32};
  }

  static TensorType RealNumberType() {
    return TensorType{DT_DOUBLE, DT_FLOAT,  DT_FLOAT16, DT_INT16,  DT_INT32, DT_INT64,
                      DT_INT8,   DT_UINT16, DT_UINT32,  DT_UINT64, DT_UINT8, DT_BF16};
  }

  static TensorType ComplexDataType() { return TensorType{DT_COMPLEX128, DT_COMPLEX64, DT_COMPLEX32}; }

  static TensorType IntegerDataType() {
    return TensorType{DT_INT16, DT_INT32, DT_INT64, DT_INT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_UINT8};
  }

  static TensorType SignedDataType() { return TensorType{DT_INT16, DT_INT32, DT_INT64, DT_INT8}; }

  static TensorType UnsignedDataType() { return TensorType{DT_UINT16, DT_UINT32, DT_UINT64, DT_UINT8}; }

  static TensorType FloatingDataType() { return TensorType{DT_DOUBLE, DT_FLOAT, DT_FLOAT16}; }

  static TensorType IndexNumberType() { return TensorType{DT_INT32, DT_INT64}; }

  static TensorType UnaryDataType() {
    return TensorType{DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_BF16, DT_COMPLEX32};
  }

  static TensorType FLOAT() { return TensorType{DT_FLOAT, DT_FLOAT16, DT_BF16}; }

  std::shared_ptr<TensorTypeImpl> tensor_type_impl_;
};
```
