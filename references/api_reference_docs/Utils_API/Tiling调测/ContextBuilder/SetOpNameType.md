# SetOpNameType

**页面ID:** atlasascendc_api_07_1015  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_1015.html

---

#### 功能说明

设置算子的名字与类型

#### 函数原型

```
ContextBuilder &SetOpNameType(const std::string& opName, const std::string& opType)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| opName | 输入 | 算子名字 |
| opType | 输入 | 算子类型 |

#### 返回值说明

当前ContextBuilder的对象。

#### 约束说明

无

#### 调用示例

```
std::string opName = "tmpNode";
std::string opType = "FlashAttentionScore";
context_ascendc::ContextBuilder builder;
(void)builder.SetOpNameType(opName, opType);
```
