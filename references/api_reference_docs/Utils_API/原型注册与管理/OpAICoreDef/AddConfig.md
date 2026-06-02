# AddConfig

**页面ID:** atlasascendc_api_07_0986  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0986.html

---

#### 功能说明

注册算子支持的AI处理器型号以及OpAICoreConfig信息。

#### 函数原型

```
void AddConfig(const char *soc)
void AddConfig(const char *soc, OpAICoreConfig &aicore_config)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| soc | 输入 | 支持的AI处理器型号。填写规则请参考算子工程目录下编译配置项文件CMakePresets.json中的ASCEND_COMPUTE_UNIT字段，该字段取值在使用msOpGen创建工程时自动生成。 |
| aicore_config | 输入 | AI Core配置信息请参考OpAICoreConfig定义。 |

#### 约束说明

不传入aicore_config参数时，对OpAICoreConfig结构中的部分参数会配置成默认值，具体的参数和默认值如下表所示：

**表1 **不传入aicore_config参数时，OpAICoreConfig默认配置

| 配置参数 | 说明 | 默认值 |
| --- | --- | --- |
| DynamicCompileStaticFlag | 用于标识该算子实现是否支持入图时的静态Shape编译。 | true |
| DynamicFormatFlag | 标识是否根据SetOpSelectFormat设置的函数自动推导算子输入输出支持的dtype和format。 | true |
| DynamicRankSupportFlag | 标识算子是否支持dynamicRanK（动态维度）。 | true |
| DynamicShapeSupportFlag | 用于标识该算子是否支持入图时的动态Shape场景。 | true |
| NeedCheckSupportFlag | 标识是否在算子融合阶段调用算子参数校验函数进行data type与Shape的校验。 | false |
| PrecisionReduceFlag | 此字段用于进行ATC模型转换或者进行网络调测时，控制算子的精度模式。 | true |
