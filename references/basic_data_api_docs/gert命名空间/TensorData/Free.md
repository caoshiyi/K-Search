# Free

**页面ID:** atlasopapi_07_00190  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00190.html

---

#### 函数功能

释放Tensor。

#### 函数原型

```
ge::graphStatus Free()
```

#### 参数说明

无。

#### 返回值说明

成功时返回：ge::GRAPH_SUCCESS，失败时返回manager函数返回的状态码。

关于ge::graphStatus类型的定义，请参见ge::graphStatus。

#### 约束说明

无。

#### 调用示例

```
auto addr = reinterpret_cast<void *>(0x10);
TensorData td(addr, nullptr);
td.Free();
```
