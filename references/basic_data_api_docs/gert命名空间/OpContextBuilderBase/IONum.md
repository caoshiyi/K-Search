# IONum

**页面ID:** atlasopapi_07_00603  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00603.html

---

#### 函数功能

设置算子输入输出IR原型个数，用于构造各子类Context的基类ExtendedKernelContext中ComputeNodeInfo信息。该接口默认每个IR原型输入输出的实例个数为1。

#### 函数原型

```
T &IONum(size_t input_ir_num, size_t output_ir_num)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| input_ir_num | 输入 | 输入IR原型个数。 |
| output_ir_num | 输入 | 输出IR原型个数。 |

#### 返回值说明

返回子类对象T类型的引用，用于支持子类链式调用。

#### 约束说明

此接口与IOInstanceNum接口互斥。仅需调用2种接口的一种即可，以先调用的接口为准。
