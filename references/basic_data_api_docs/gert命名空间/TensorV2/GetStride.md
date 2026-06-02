# GetStride

**页面ID:** atlasopapi_07_00765  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00765.html

---

#### 函数功能

获取Tensor中的stride。

#### 函数原型

```
const Stride &GetStride() const
```

#### 参数说明

无

#### 返回值说明

只读的Tensor Stride引用。关于Stride类型的定义，请参见Stride。

#### 约束说明

无

#### 调用示例

```
StorageShape sh({1, 2, 3}, {1, 2, 3});
Stride str({6, 3, 1});
TensorV2 t = {sh, {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}}, kOnHost, ge::DT_FLOAT, nullptr,nullptr, str, 0};
auto stride = t.GetStride(); // str
```
