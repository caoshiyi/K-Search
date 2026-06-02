# MutableStride

**页面ID:** atlasopapi_07_00767  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00767.html

---

#### 函数功能

获取Tensor中的Stride引用，该Stride可修改。

#### 函数原型

```
Stride &MutableStride()
```

#### 参数说明

无

#### 返回值说明

可写的Tensor Stride引用。

关于Stride类型的定义，请参见Stride。

#### 约束说明

无

#### 调用示例

```
StorageShape sh({1, 2, 3}, {1, 2, 3});
Stride str({6, 3, 1});
TensorV2 t = {sh, {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}}, kOnHost, ge::DT_FLOAT, nullptr,nullptr, str, 0};
t.MutableStride() = Stride({6, 1, 1});
auto stride = t.GetStride(); // Stride({6, 1, 1})
```
