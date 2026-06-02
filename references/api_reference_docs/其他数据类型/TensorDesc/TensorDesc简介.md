# TensorDesc简介

**页面ID:** atlasascendc_api_07_00062  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_00062.html

---

TensorDesc用于储存ListTensorDesc.GetDesc()中根据index获取对应的Tensor描述信息。

#### 原型定义

```
template<class T> class TensorDesc {
    TensorDesc();
    ~TensorDesc();
    void SetShapeAddr(uint64_t* shapePtr);
    uint64_t GetDim();
    uint64_t GetIndex();
    uint64_t GetShape(uint32_t offset);
    T* GetDataPtr();
    GlobalTensor<T> GetDataObj();
}
```

#### 模板参数

**表1 **模板参数说明

| 参数名 | 描述 |
| --- | --- |
| T | Tensor数据类型。 |

#### 成员函数

```
TensorDesc()
~TensorDesc()
void SetShapeAddr(uint64_t* shapePtr)
uint64_t GetDim()
uint64_t GetIndex()
uint64_t GetShape(uint32_t offset)
T* GetDataPtr()
GlobalTensor<T> GetDataObj()
```
