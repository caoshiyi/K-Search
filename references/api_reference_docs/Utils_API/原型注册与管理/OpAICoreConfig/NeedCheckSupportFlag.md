# NeedCheckSupportFlag

**页面ID:** atlasascendc_api_07_0995  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0995.html

---

#### 功能说明

标识是否在算子融合阶段调用算子参数校验函数进行data type与shape的校验。

- 若配置为"true"，框架会调用通过SetCheckSupport设置的算子参数校验函数，检查算子是否支持指定输入，此场景下需要自行实现算子参数校验的回调函数。
- 若配置为"false"，表示不需要进行校验。

#### 函数原型

```
OpAICoreConfig &NeedCheckSupportFlag(bool flag)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| flag | 输入 | 标识是否在算子融合阶段调用算子参数校验函数进行data type与shape的校验。 |

#### 返回值说明

OpAICoreConfig类，请参考OpAICoreConfig。

#### 约束说明

无

#### 调用示例

请参考SetCheckSupport节调用示例。
