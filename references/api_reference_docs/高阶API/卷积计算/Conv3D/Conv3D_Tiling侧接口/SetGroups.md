# SetGroups

**页面ID:** atlasascendc_api_07_10095  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_10095.html

---

#### 功能说明

设置分组卷积的分组大小。分组大小为1表示普通卷积。**当前Conv3D 高阶API不支持分组卷积**。

#### 函数原型

```
void SetGroups(int64_t groups)
```

#### 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| groups | 输入 | 当前仅支持取值为1，暂不支持分组卷积。 |

#### 约束说明

在调用GetTiling接口前，本接口可选调用。若未调用本接口，默认groups=1，当前仅支持输入groups值配置为1，group>1的卷积能力暂不支持。

#### 调用示例

```
// 实例化Conv3D Api
auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
Conv3dTilingApi::Conv3dTiling conv3dApiTiling(ascendcPlatform);
conv3dApiTiling.SetGroups(groups);
```
