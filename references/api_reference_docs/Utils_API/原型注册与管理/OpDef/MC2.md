# MC2

**页面ID:** atlasascendc_api_07_0954  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0954.html

---

#### 功能说明

注册该算子是通算融合算子，调用该接口后会返回一个OpMC2Def结构，后续可通过该结构配置通信域名称。

#### 函数原型

```
OpMC2Def &MC2(void)
```

#### 参数说明

无

#### 返回值说明

OpMC2Def结构，后续可通过该结构配置通信域名称。

#### 约束说明

基于旧版本CANN包（不支持MC2特性）生成的自定义算子工程，无法兼容MC2接口。在使用非当前版本CANN包生成的自定义算子工程时，需特别注意兼容性问题。您可以通过查看自定义算子工程下cmake/util/ascendc_impl_build.py中有无_build_mc2_ctx字段来确认当前工程是否支持该特性，如果未找到该字段，则需要重新生成自定义算子工程以启用MC2特性。
