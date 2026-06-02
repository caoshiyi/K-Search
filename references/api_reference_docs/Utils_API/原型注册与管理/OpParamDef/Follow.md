# Follow

**页面ID:** atlasascendc_api_07_0974  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0974.html

---

#### 功能说明

用于指定当前输入/输出的datatype/format/shape信息与之前定义过的某个输入一致。

#### 函数原型

- datatype/format/shape全部信息保持一致

```
OpParamDef &Follow(const char *paramName)
```

- datatype/format/shape指定信息保持一致

```
OpParamDef &Follow(const char *paramName, FollowType ftype)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| paramName | 输入 | 之前定义过的输入名。 |
| ftype | 输入 | ftype类型为枚举类FollowType，表示Follow的模式，取值如下： - ALL，datatype/format/shape与paramName均保持一致；- DTYPE，datatype与paramName保持一致；- FORMAT，format与paramName保持一致；- SHAPE，shape与paramName保持一致。 |

#### 返回值说明

OpParamDef算子定义，OpParamDef请参考OpParamDef。

#### 约束说明

- 所有Follow的数据源头仅支持为Input。
- 针对shape推导，仅支持输出参数Follow输入参数，不支持输入参数Follow另一个输入参数。
- 可以支持链式Follow，例如C Follow B，B Follow A，但此时Follow的模式不可中途变更（ftype需要保持一致）。
- 使用Follow接口通常比InferShape函数逻辑更加简单，能用Follow表达的逻辑，建议使用Follow接口，则无需再编写注册InferShape函数。
- InferShape推导函数和使用Follow接口去Follow shape不能混用，即不支持部分输出采用Infershape推导、部分输出采用Follow推导的情况。若用户同时使用了InferShape函数和Follow接口，以用户的InferShape函数为准，需要保证在InferShape函数中能够推导出所有的输出shape。
- datatype/format同时支持Follow输入的参数类型为DataTypeList/FormatList ，调用Follow后，当前输入/输出的datatype/format与paramName组合后的datatype/format一致。

#### 调用示例

- 输出“y1”Follow输入“x1”场景，此时“y1”的datatype、format以及shape都将会和“x1”保持一致。

```
this->Input("x1")
    .ParamType(REQUIRED)
    .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
    .Format({ge::FORMAT_ND, ge::FORMAT_ND});
this->Input("x2")
    .ParamType(REQUIRED)
    .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
    .Format({ge::FORMAT_ND, ge::FORMAT_ND});
this->Output("y1")
    .ParamType(REQUIRED)
    .Follow("x1");
```

- 链式follow场景，y1 -> x2 -> x1。此时“y1”的datatype、format以及shape都将会和“x1”保持一致，“x2”的datatype、format和“x1”保持一致。

```
this->Input("x1")
    .ParamType(REQUIRED)
    .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
    .Format({ge::FORMAT_ND, ge::FORMAT_ND});
this->Input("x2")
    .ParamType(REQUIRED)
    .Follow("x1");
this->Output("y1")
    .ParamType(REQUIRED)
    .Follow("x2");
```
