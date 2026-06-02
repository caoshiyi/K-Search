# GetCapacity

**页面ID:** atlasopapi_07_00254  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00254.html

---

#### 函数功能

获取本实例可容纳的最大tiling data长度。

#### 函数原型

**size_t GetCapacity() const**

#### 参数说明

无。

#### 返回值说明

最大tiling data长度。

#### 约束说明

无。

#### 调用示例

```
auto td_buf = TilingData::CreateCap(100U);
auto td = reinterpret_cast<TilingData *>(td_buf.get());
size_t cap = td->GetCapacity(); // 100U
```
