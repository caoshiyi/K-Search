# AppendDim

**页面ID:** atlasopapi_07_00161  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00161.html

---

#### 函数功能

向后扩展一个dim值，如果扩展的dim数量超出Shape的最大限制，那么本函数不做任何事情。

#### 函数原型

```
Shape& AppendDim(const int64_t value)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| value | 输入 | 扩展的dim值。 |

#### 返回值说明

this引用。

#### 约束说明

无。

#### 调用示例

```
Shape shape0({3, 256, 256});
shape0.AppendDim(1024); // 3,256,256,1024
```
