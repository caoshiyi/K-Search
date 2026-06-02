# ShareFrom

**页面ID:** atlasopapi_07_00193  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00193.html

---

#### 函数功能

使当前的TensorData对象共享另一个对象的内存以及内存管理函数。

#### 函数原型

```
ge::graphStatus ShareFrom(const TensorData &other)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| other | 输入 | 另一个TensorData对象。 |

#### 返回值说明

成功时返回ge::GRAPH_SUCCESS。

#### 约束说明

无。

#### 调用示例

```
auto addr = reinterpret_cast<void *>(0x10);
TensorData td1(addr, HostAddrManager, 100U, kOnHost);
TensorData td2(addr, nullptr);
td2.ShareFrom(td1);
```
