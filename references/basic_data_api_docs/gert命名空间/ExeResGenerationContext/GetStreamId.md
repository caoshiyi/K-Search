# GetStreamId

**页面ID:** atlasopapi_07_00697  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00697.html

---

#### 函数功能

返回当前算子绑定的StreamId。

#### 函数原型

```
int64_t GetStreamId() const
```

#### 参数说明

无

#### 返回值说明

返回当前算子绑定的StreamId。

#### 约束说明

只有在每个算子只能绑定一个Stream的前提下可以使用该接口。

#### 调用示例

```
int64_t GetStreamId(ExeResGenerationContext* context) {
  auto streamId= context->GetStreamId();
  ...
}
```
