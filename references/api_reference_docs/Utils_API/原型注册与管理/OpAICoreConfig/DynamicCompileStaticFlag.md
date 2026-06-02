# DynamicCompileStaticFlag

**页面ID:** atlasascendc_api_07_0991  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0991.html

---

#### 功能说明

用于标识该算子实现是否支持入图时的静态Shape编译。

#### 函数原型

```
OpAICoreConfig &DynamicCompileStaticFlag(bool flag)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| flag | 输入 | 用户开发的自定义算子，如果需要支持入图时静态Shape场景下的编译，需要配置该选项为true，否则配置为false。 |

#### 返回值说明

OpAICoreConfig类，请参考OpAICoreConfig。

#### 约束说明

无
