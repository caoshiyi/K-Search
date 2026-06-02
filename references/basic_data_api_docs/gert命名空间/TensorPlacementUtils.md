# TensorPlacementUtils

**页面ID:** atlasopapi_07_00194  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00194.html

---

#### 函数功能

提供一组函数，判断TensorPlacement的位置。

#### 函数原型

```
class TensorPlacementUtils {
 public:
  // 判断Tensor是否位于Device上的内存
  static bool IsOnDevice(TensorPlacement placement) {
    ...
  }
  // 判断Tensor是否位于Host上
  static bool IsOnHost(TensorPlacement placement) {
    ...
  }
  // 判断Tensor是否位于Host上，且数据紧跟在结构体后面
  static bool IsOnHostFollowing(TensorPlacement placement) {
    ...
  }
  // 判断Tensor是否位于Host上，且数据不紧跟在结构体后面
  static bool IsOnHostNotFollowing(TensorPlacement placement) {
    ...
  }
  // 判断Tensor是否位于Device上的内存
  static bool IsOnDeviceHbm(TensorPlacement placement) {
    ...
  }
  // 判断Tensor是否位于Device上的P2p内存
  static bool IsOnDeviceP2p(TensorPlacement placement) {
    ...
  }
};
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| placement | 输入 | 需要进行判断的TensorPlacement枚举。 |

#### 返回值说明

true表示是；false表示不是。

#### 约束说明

无。

#### 调用示例

```
TensorData tensor_data;
tensor_data.SetPlacement(TensorPlacement::kOnHost);
auto on_host = TensorPlacementUtils::IsOnHost(tensor_data.GetPlacement()); // on_host is true
```
