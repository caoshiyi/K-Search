# GetAttachedStreamInfos

**页面ID:** atlasopapi_07_00696  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00696.html

---

#### 函数功能

返回当前算子绑定的所有StreamInfo列表。

#### 函数原型

```
std::vector<StreamInfo> GetAttachedStreamInfos() const
```

#### 参数说明

无

#### 返回值说明

返回当前算子绑定的所有StreamInfo列表。

#### 约束说明

无

#### 调用示例

```
std::vector<StreamInfo> GetAttachedStreamInfos(ExeResGenerationContext* context) {
  std::vector<StreamInfo> streamInfoList = context->GetAttachedStreamInfos();
  ...
}
```
