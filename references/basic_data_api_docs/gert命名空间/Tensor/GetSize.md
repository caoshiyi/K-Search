# GetSize

**页面ID:** atlasopapi_07_00201  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00201.html

---

#### 函数功能

获取Tensor数据的内存大小。

#### 函数原型

**size_t GetSize() const**

#### 参数说明

无。

#### 返回值说明

内存大小，单位是字节。

#### 约束说明

无。

#### 调用示例

```
StorageShape sh({1, 2, 3}, {1, 2, 3});
Tensor t = {sh, {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}}, kOnHost, ge::DT_FLOAT, nullptr};
auto td_size = t.GetSize(); // 1*2*3*sizeof(float) = 24;
```
