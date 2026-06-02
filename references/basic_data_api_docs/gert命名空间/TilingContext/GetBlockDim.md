# GetBlockDim

**页面ID:** atlasopapi_07_00237  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00237.html

---

#### 函数功能

获取blockDim，即参与计算的Vector或者Cube核数。blockDim的详细概念和设置方式请参考SetBlockDim。

#### 函数原型

**uint32_t GetBlockDim() const**

#### 参数说明

无。

#### 返回值说明

返回blockDim。

#### 约束说明

无。

#### 调用示例

```
ge::graphStatus Tiling4XXX(TilingContext* context) {
  auto block_dim = context->GetBlockDim();
  // ...
}
```
