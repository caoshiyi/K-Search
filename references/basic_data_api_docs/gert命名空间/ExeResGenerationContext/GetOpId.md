# GetOpId

**页面ID:** atlasopapi_07_00702  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00702.html

---

#### 函数功能

获取当前算子的唯一标识ID。

#### 函数原型

```
int64_t GetOpId() const
```

#### 参数说明

无

#### 返回值说明

返回一个int64_t类型的整数，代表这个算子在计算图中的唯一编号。

#### 约束说明

无

#### 调用示例

```
int64_t GetOpId(ExeResGenerationContext* context) {
  auto opId= context->GetOpId();
  // ...
}
```
