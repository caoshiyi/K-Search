# NeedAtomic

**页面ID:** atlasopapi_07_00241  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00241.html

---

#### 函数功能

获取是否需要atomic clean的标识值。

#### 函数原型

**bool NeedAtomic() const**

#### 参数说明

无。

#### 返回值说明

- true：代表需要做atomic clean。
- false：代表不需要做atomic clean。

#### 约束说明

无。

#### 调用示例

```
ge::graphStatus Tiling4XXX(TilingContext* context) {
  bool is_need_atomic = context->NeedAtomic();
  // ...
}
```
