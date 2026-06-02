# TensorPlacement

**页面ID:** atlasopapi_07_00273  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00273.html

---

```
enum TensorPlacement {
    kOnDeviceHbm,  ///< Tensor位于Device
    kOnHost,       ///< Tensor位于Host
    kFollowing,    ///< Tensor位于Host，且数据紧跟在结构体后面
    kOnDeviceP2p,  ///< Tensor位于Device, P2p内存指的是Device内存透到PCIE BAR空间上可以让NPU跨PCIE访问的地址空间
    kTensorPlacementEnd
};
```
