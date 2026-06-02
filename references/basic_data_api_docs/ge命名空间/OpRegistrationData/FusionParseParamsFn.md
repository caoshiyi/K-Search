# FusionParseParamsFn

**页面ID:** atlasopapi_07_00391  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00391.html

---

#### 函数功能

注册解析融合算子属性的函数。

#### 函数原型

```
OpRegistrationData &FusionParseParamsFn(const FusionParseParamFunc &fusionParseParamFn)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| fusionParseParamFn | 输入 | 解析融合算子属性的函数，请参见回调函数FusionParseParamFunc。 |

#### 约束说明

对于融合算子插件，FusionParseParamsFn接口后续版本将会废弃，请使用FusionParseParamsFn（Overload）接口进行融合算子属性的解析。

#### 回调函数FusionParseParamFunc

用户自定义并实现FusionParseParamFunc类函数，完成原始模型中属性到适配昇腾AI处理器的模型中属性的映射，将结果填到Operator类中。

```
Status FusionParseParamFunc(const  vector<const google::protobuf::Message *> &v_op_origin, ge::Operator  &op_dest)
```

**表1 **参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| v_op_origin | 输入 | 一组scope内的protobuf格式的数据结构（来源于原始模型的prototxt文件），包含算子属性信息。 |
| op_dest | 输出 | 融合算子数据结构，保存融合算子信息。          关于Operator类，请参见Operator。 |
