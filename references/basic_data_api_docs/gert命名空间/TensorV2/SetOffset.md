# SetOffset

**页面ID:** atlasopapi_07_00768  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00768.html

---

#### 函数功能

设置Tensor中的offset。

#### 函数原型

```
void SetOffset(const int64_t offset)
```

#### 参数说明

Tensor的offset。

#### 约束说明

无

#### 调用示例

```
StorageShape sh({1, 2, 3}, {1, 2, 3});
Stride str({6, 3, 1});
TensorV2 t = {sh, {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}}, kOnHost, ge::DT_FLOAT, nullptr,nullptr, str, 0};
t.SetOffset(10);
auto offset = t.GetOffset(); // 10
```
