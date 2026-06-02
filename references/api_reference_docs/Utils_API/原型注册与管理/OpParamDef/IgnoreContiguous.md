# IgnoreContiguous

**页面ID:** atlasascendc_api_07_0965  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0965.html

---

#### 功能说明

某些算子支持非连续的tensor，在算子的实现中对非连续的tensor做了转换处理。配置该参数后，框架会忽略对非连续的校验。

#### 函数原型

```
OpParamDef &IgnoreContiguous(void)
```

#### 参数说明

无

#### 返回值说明

OpParamDef算子定义，OpParamDef请参考OpParamDef。

#### 约束说明

无
