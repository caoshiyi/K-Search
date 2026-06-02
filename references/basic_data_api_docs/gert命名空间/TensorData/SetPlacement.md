# SetPlacement

**页面ID:** atlasopapi_07_00189  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00189.html

---

#### 函数功能

设置tensor的placement。

#### 函数原型

```
void SetPlacement(const TensorPlacement placement)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| placement | 输入 | tensor的placement。 关于TensorPlacement类型的定义，请参见TensorPlacement。 |

#### 返回值说明

无。

#### 约束说明

无。

#### 调用示例

```
auto addr = reinterpret_cast<void *>(0x10);
TensorData td(addr, nullptr);
auto td_place = td.SetPlacement(kOnHost);
```
