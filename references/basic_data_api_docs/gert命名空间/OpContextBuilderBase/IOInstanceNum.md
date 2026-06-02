# IOInstanceNum

**页面ID:** atlasopapi_07_00604  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00604.html

---

#### 函数功能

当输入IR原型实例个数不为1时（一般是可选输入或动态输入场景），需要设置算子每个输入IR原型的实例个数，用于构造各子类Context的基类ExtendedKernelContext中ComputeNodeInfo信息。

#### 函数原型

```
T &IOInstanceNum(const std::vector<uint32_t> &input_instance_num, const std::vector<uint32_t> &output_instance_num)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| input_instance_num | 输入 | 是一个vector数组输入，vector的size代表算子原型输入个数，vector数组上每个位置的数字代表每个IR原型输入的实例个数。 |
| output_instance_num | 输入 | 是一个vector数组输入，vector的size代表算子原型输出个数，vector数组上每个位置的数字代表每个IR原型输出的实例个数。 |

#### 返回值说明

返回子类对象T类型的引用，用于支持子类链式调用。

#### 约束说明

此接口与IONum接口互斥。仅需调用2种接口的一种即可，以先调用的接口为准。
