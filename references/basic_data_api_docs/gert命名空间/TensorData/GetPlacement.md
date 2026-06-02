# GetPlacement

**页面ID:** atlasopapi_07_00188  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00188.html

---

#### 函数功能

获取tensor的placement，tensor数据所在的设备位置。

#### 函数原型

```
TensorPlacement GetPlacement() const
```

#### 参数说明

无。

#### 返回值说明

tensor的placement。关于TensorPlacement类型的定义，请参见TensorPlacement。

#### 约束说明

无。

#### 调用示例

```
auto addr = reinterpret_cast<void *>(0x10);
TensorData td(addr, HostAddrManager, 100U, kOnHost);
auto td_place = td.GetPlacement(); // kOnHost
```
