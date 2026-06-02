# SetDataSize

**页面ID:** atlasopapi_07_00256  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00256.html

---

#### 函数功能

设置tiling data长度。

#### 函数原型

**void SetDataSi****ze(const size_t size)**

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| size | 输入 | tiling data长度。 |

#### 返回值说明

无。

#### 约束说明

无。

#### 调用示例

```
auto td_buf = TilingData::CreateCap(100U);
auto td = reinterpret_cast<TilingData *>(td_buf.get());
size_t data_size = td->GetDataSize(); // 0

td->SetDataSize(100U);
data_size = td->GetDataSize(); // 100
```
