# TilingData结构体

**页面ID:** atlasascendc_api_07_10048  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_10048.html

---

#### 功能说明

在算子实现中，由Tiling组装通信配置项，核函数实现时将Tiling配置项通过入参传递给Kernel侧通信API做通信计算。本节TilingData结构体包括Mc2InitTiling和Mc2CcTiling，这两个结构体均通过调用GetTiling接口返回。其中，Mc2CcTiling为具体每个通信任务的参数配置，当算子中有多个通信任务时，可定义多个Mc2CcTiling参数（最多支持定义8个）。

#### 参数说明

**表1 **Mc2InitTiling参数说明

| 参数名 | 描述 |
| --- | --- |
| reserved | 初始化通信任务配置。uint8_t *类型，支持最大长度64字节，该结构体仅支持通过接口GetTiling获取。 |

**表2 **Mc2CcTiling参数说明

| 参数名 | 描述 |
| --- | --- |
| reserved | 各通信域中每个通信任务的参数配置。uint8_t *类型，支持最大长度280字节，该结构体仅支持通过接口GetTiling获取。注意，最多支持配置8个通信任务。 |
