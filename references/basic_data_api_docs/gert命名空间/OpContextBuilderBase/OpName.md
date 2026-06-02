# OpName

**页面ID:** atlasopapi_07_00602  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00602.html

---

#### 函数功能

设置算子名称，用于构造各子类Context的基类ExtendedKernelContext中ComputeNodeInfo信息。

#### 函数原型

```
T &OpName(const ge::AscendString &op_name)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| op_name | 输入 | 算子名称。 |

#### 返回值说明

返回子类对象T类型的引用，用于支持子类链式调用。

#### 约束说明

无
