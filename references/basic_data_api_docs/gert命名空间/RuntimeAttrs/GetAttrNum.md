# GetAttrNum

**页面ID:** atlasopapi_07_00148  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00148.html

---

#### 函数功能

获取属性的数量。

#### 函数原型

```
size_t GetAttrNum() const
```

#### 参数说明

无。

#### 返回值说明

属性的数量。

#### 约束说明

无。

#### 调用示例

```
const RuntimeAttrs * runtime_attrs = kernel_context->GetAttrs();
size_t attr_num = runtime_attrs->GetAttrNum();
```
