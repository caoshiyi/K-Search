# MutableTensorData

**页面ID:** atlasopapi_07_00763  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00763.html

---

#### 函数功能

获取Tensor中的数据。

#### 函数原型

```
TensorData &MutableTensorData()
```

#### 参数说明

无

#### 返回值说明

可写的tensor data引用。

关于TensorData类型的定义，请参见TensorData。

#### 约束说明

无

#### 调用示例

```
StorageShape sh({1, 2, 3}, {1, 2, 3});
TensorV2 t = {sh, {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}}, kOnHost, ge::DT_FLOAT, nullptr};
auto a = reinterpret_cast<void *>(10);
t.MutableTensorData() = TensorData{a, nullptr}; // 设置新tensordata
auto addr = t.GetTensorData().GetAddr(); // 10
```
