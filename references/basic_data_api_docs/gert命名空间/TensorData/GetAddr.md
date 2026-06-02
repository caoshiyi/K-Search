# GetAddr

**页面ID:** atlasopapi_07_00185  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00185.html

---

#### 函数功能

获取tensor数据地址。若存在manager函数，则由manager函数给出地址。

#### 函数原型

```
TensorAddress GetAddr() const
```

#### 参数说明

无。

#### 返回值说明

tensor地址。

#### 约束说明

无。

#### 调用示例

```
auto addr = reinterpret_cast<void *>(0x10);
TensorData td(addr, nullptr);
auto addr = td.GetAddr(); // 0x10
```
