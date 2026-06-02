# Scalar

**页面ID:** atlasascendc_api_07_0967  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0967.html

---

#### 功能说明

配置该参数后，自动生成的单算子API（aclnnxxx）接口中，输入类型为aclScalar类型。

#### 函数原型

```
OpParamDef &Scalar()
```

#### 参数说明

无

#### 返回值说明

OpParamDef算子定义，OpParamDef请参考OpParamDef。

#### 约束说明

- 仅支持对算子输入做该参数配置，如果对算子输出配置该参数，则配置无效。
- 该接口仅在如下场景支持：

  - 通过单算子API执行的方式开发单算子调用应用。
  - 间接调用单算子API(aclnnxxx)接口：Pytorch框架单算子直调的场景。

#### 调用示例

```
this->Input("x")
    .Scalar()
```
