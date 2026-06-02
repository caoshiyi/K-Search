# FusionParseParamsFn（Overload）

**页面ID:** atlasopapi_07_00392  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00392.html

---

#### 函数功能

注册解析融合算子属性的函数，为FusionParseParamsFn的重载函数。

#### 函数原型

```
OpRegistrationData &FusionParseParamsFn(const FusionParseParamByOpFunc &fusion_parse_param_fn)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| **fusion_parse_param_fn** | 输入 | 解析融合算子属性的函数，请参见回调函数FusionParseParamByOpFunc。 |

#### 回调函数FusionParseParamByOpFunc

用户自定义并实现FusionParseParamByOpFunc类函数，完成原始模型中属性到适配昇腾AI处理器的模型中的属性映射，将结果填到Operator类中。

```
Status FusionParseParamByOpFunc(const std::vector<ge::Operator> &op_src,  ge::Operator &op_dest)
```

**表1 **参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| op_src | 输入 | 一组scope内存储原始模型中算子属性的融合算子数据结构。          关于Operator类，请参见Operator。 |
| op_dest | 输出 | 融合算子数据结构，保存融合算子信息。          关于Operator类，请参见Operator。 |

#### 调用示例

```
REGISTER_CUSTOM_OP(XXXXXX)
.FrameworkType(TENSORFLOW)
.FusionParseParamsFn(FusionParseParamsFn)
.OriginOpType(XXXXX)
.ImplyType(XXXXX);
```
