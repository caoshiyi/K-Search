# TensorVersion

**页面ID:** atlasopapi_07_00770  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00770.html

---

头文件位于CANN软件安装后文件存储路径下的include/exe_graph/runtime/tensor.h。

```
enum TensorVersion : uint8_t {
  kTensorV1 = 0, ///< 不携带非连续描述信息
  kTensorV2 = 1, ///< 携带非连续描述信息
};
```
