# SetAicpuBlockDim

**页面ID:** atlasopapi_07_00589  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00589.html

---

#### 函数功能

设置可以调度的AI CPU核数。

在使用HCCL高阶API的BatchWrite接口时，可以通过合理设置AI CPU核数获得更好的性能。

#### 函数原型

**ge::graphStatus SetAicpuBlockDim(uint32_t block_dim)**

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| block_dim | 输入 | 可以调度的AI CPU核数。 |

#### 返回值说明

设置成功时返回“ge::GRAPH_SUCCESS”。

关于graphStatus的定义，请参见ge::graphStatus。

#### 约束说明

无。

#### 调用示例

```
ge::graphStatus Tiling4XXX(TilingContext* context) {
  auto ret = context->SetAicpuBlockDim(5U);
  // ...
}
```
