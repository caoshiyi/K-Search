# ValueDepend

**页面ID:** atlasascendc_api_07_0964  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0964.html

---

#### 功能说明

标识该输入是否为“数据依赖输入”，数据依赖输入是指在Tiling/InferShape等函数实现时依赖该输入的具体数据。该输入数据为host侧数据，开发者在Tiling函数/InferShape函数中可以通过TilingContext类的GetInputTensor/InferShapeContext类的GetInputTensor获取这个输入数据。

#### 函数原型

```
OpParamDef &ValueDepend(Option value_depend)
OpParamDef &ValueDepend(Option value_depend, DependScope scope)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| value_depend | 输入 | value_depend有以下两种取值：                     - REQUIRED：表示算子的输入必须是Const类型。            在调用算子的SetCheckSupport时，会校验算子的输入是否是Const类型。若校验通过，则将此输入的值下发到算子；否则报错。           - OPTIONAL：表示算子的输入可以是Const类型，也可以不是Const类型。如果输入是Const类型，则将输入的值下发到算子，否则不下发。 |
| scope | 输入 | scope类型为枚举类型DependScope，支持的取值为：                     - ALL：指在Tiling/InferShape等函数实现时都依赖该输入的具体数据，行为与调用单参数ValueDepend重载接口一致。           - TILING：指仅在Tiling时依赖Tensor的值，可以支持Tiling下沉。 |

#### 返回值说明

OpParamDef算子定义，OpParamDef请参考OpParamDef。

#### 约束说明

仅支持对算子输入配置，且仅支持输入的参数数据类型配置为DT_INT64/DT_FLOAT/DT_BOOL，对应生成的输出类型分别为aclIntArray、aclFloatArray和aclBoolArray（aclnn数据类型）。
