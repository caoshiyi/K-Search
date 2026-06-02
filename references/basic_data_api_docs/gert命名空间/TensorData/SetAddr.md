# SetAddr

**页面ID:** atlasopapi_07_00191  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00191.html

---

#### 函数功能

设置tensor地址。

#### 函数原型

```
ge::graphStatus SetAddr(const ConstTensorAddressPtr addr, TensorAddrManager manager)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| tensor地址。 ``` using ConstTensorAddressPtr = const void *; ``` |  |  |
| tensor的管理函数。 ``` using TensorAddrManager = ge::graphStatus (*)(TensorAddress addr, TensorOperateType operate_type, void **out); ``` |  |  |

#### 返回值说明

成功时返回ge::GRAPH_SUCCESS；失败时返回manager管理函数中定义的错误码。

#### 约束说明

无。

#### 调用示例

```
auto addr = reinterpret_cast<void *>(0x10);
TensorData td(addr, nullptr);
td.SetAddr(addr, HostAddrManager);
```
