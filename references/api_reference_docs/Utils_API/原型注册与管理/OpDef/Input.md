# Input

**页面ID:** atlasascendc_api_07_0947  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0947.html

---

#### 功能说明

注册算子输入，调用该接口后会返回一个OpParamDef结构，后续可通过该结构配置算子输入信息。

#### 函数原型

```
OpParamDef &Input(const char *name)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| name | 输入 | 算子输入名称。 |

#### 返回值说明

算子参数定义，OpParamDef实例，具体请参考OpParamDef。

#### 约束说明

参数注册的顺序需要和算子kernel入口函数一致。
