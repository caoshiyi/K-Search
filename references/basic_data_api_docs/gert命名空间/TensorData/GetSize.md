# GetSize

**页面ID:** atlasopapi_07_00186  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00186.html

---

#### 函数功能

获取tensor数据的内存大小。

#### 函数原型

```
size_t GetSize() const
```

#### 参数说明

无。

#### 返回值说明

tensor所占内存大小，单位为字节。

#### 约束说明

无。

#### 调用示例

```
auto addr = reinterpret_cast<void *>(0x10);
TensorData td(addr, HostAddrManager, 100U, kOnHost);
auto td_size = td.GetSize(); // 100U
```
