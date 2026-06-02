# SetOriginFormat

**页面ID:** atlasopapi_07_00755  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00755.html

---

#### 函数功能

设置Tensor的原始数据格式。

#### 函数原型

```
void SetOriginFormat(const ge::Format origin_format)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| origin_format | 输入 | 原始数据格式。 关于ge::Format类型的定义，请参见Format。 |

#### 约束说明

无

#### 调用示例

```
StorageShape sh({1, 2, 3}, {1, 2, 3});
TensorV2 t = {sh, {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}}, kOnHost, ge::DT_FLOAT, nullptr};
t.SetOriginFormat(ge::FORMAT_NHWC);
t.SetStorageFormat(ge::FORMAT_NC1HWC0);
auto fmt = t.GetOriginFormat(); // ge::FORMAT_NHWC
```
