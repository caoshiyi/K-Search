# To

**页面ID:** atlasascendc_api_07_0969  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0969.html

---

#### 功能说明

配置该参数后，在调用单算子API(aclnnxxx)接口时，会将aclScalar/aclScalarList的数据类型转换为该参数指定的数据类型。

指定的数据类型可以通过两种方式传入：

- 直接传入datatype；
- 传入输入的名称，表示数据类型和该输入的datatype保持一致。

**该接口仅在如下场景支持****：**

- 通过单算子API执行的方式开发单算子调用应用。
- 间接调用单算子API(aclnnxxx)接口：Pytorch框架单算子直调的场景。

#### 函数原型

```
OpParamDef &To(const ge::DataType type)
OpParamDef &To(const char *name)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| **type** | 输入 | 指定的数据类型。 |
| **name** | 输入 | 算子输入的名称，表示指定的数据类型和该输入的数据类型一致。 |

#### 返回值说明

OpParamDef算子定义，OpParamDef请参考OpParamDef。

#### 约束说明

无

#### 调用示例

```
this->Input("x")
    .ParamType(REQUIRED)
    .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
    .Format({ge::FORMAT_ND, ge::FORMAT_ND})
    .ScalarList()
    .To(ge::DT_FLOAT);
this->Input("x1")
    .ParamType(REQUIRED)
    .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
    .Format({ge::FORMAT_ND, ge::FORMAT_ND});
this->Input("x2")
    .ParamType(REQUIRED)
    .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
    .Format({ge::FORMAT_ND, ge::FORMAT_ND})
    .ScalarList()
    .To("x1");
```
