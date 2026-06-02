# Release

**页面ID:** atlasopapi_07_00790  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00790.html

---

#### 函数功能

释放对TensorAddress的所有权，本接口调用后，本对象不再管理TensorAddress，而且TensorAddress并没有被释放，因此调用者负责通过manager释放TensorAddress。

#### 函数原型

```
TensorAddress Release(TensorAddrManager &manager)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| manager | 输出 | tensor的管理函数。用于管理返回的TensorAddress的函数，若本对象无所有权，那么manager被设置为nullptr。 |

#### 返回值说明

本对象持有的TensorAddress指针，若本对象不持有任何指针，则返回nullptr。

#### 约束说明

无

#### 调用示例

```
auto addr = reinterpret_cast<void *>(0x10);
TensorData td(addr, HostAddrManager, 100U, kOnHost);
TensorAddrManager NewHostAddrManager = nullptr;
TensorAddress ta = td.Release(NewHostAddrManager);
```
