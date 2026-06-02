# Update

**页面ID:** atlasopapi_07_00456  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00456.html

---

#### 函数功能

更新TensorDesc的Shape、Format、DataType属性。

#### 函数原型

```
void Update(const Shape &shape, Format format = FORMAT_ND, DataType dt = DT_FLOAT)
```

#### 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| shape | 输入 | 需刷新的Shape对象。 |
| format | 输入 | 需刷新的Format对象，默认取值FORMAT_ND。 |
| dt | 输入 | 需刷新的DataType对象，默认取值DT_FLOAT。 |

#### 返回值

无。

#### 异常处理

无。

#### 约束说明

无。
