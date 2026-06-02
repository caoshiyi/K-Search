# GetLocalMemorySize

**页面ID:** atlasopapi_07_00251  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00251.html

---

#### 函数功能

算子获取所需的Local Memory大小。

**该接口为预留接口，为后续功能做保留，不建议开发者使用，开发者无需关注。**

#### 函数原型

**uint32_t GetLocalMemorySize()**

#### 参数说明

无。

#### 返回值说明

返回Local Memory大小，如果之前没有调用SetLocalMemorySize进行设置，则返回0。

#### 约束说明

无。

#### 调用示例

```
ge::graphStatus Tiling4XXX(TilingContext* context) {
  auto local_memory_size = context->GetLocalMemorySize();
  // ...
}
```
