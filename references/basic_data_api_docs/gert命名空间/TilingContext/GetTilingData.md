# GetTilingData

**页面ID:** atlasopapi_07_00242  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00242.html

---

#### 函数功能

获取有类型的tiling data指针。

#### 函数原型

**template<typename T> auto GetTilingData() -> T***

#### 参数说明

| 参数 | 说明 |
| --- | --- |
| T | tiling data类型，sizeof(T)不可以大于编译结果中指定的最大tiling data长度。 |

#### 返回值说明

tiling data指针，失败时返回空指针。

#### 约束说明

sizeof(T)不可以大于编译结果中指定的最大tiling data长度。

#### 调用示例

```
ge::graphStatus Tiling4XXX(TilingContext* context) {
  auto tiling_data = context->GetTilingData<int64_t>();
  // ...
}
```
