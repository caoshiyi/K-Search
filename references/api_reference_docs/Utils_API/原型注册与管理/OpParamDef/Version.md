# Version

**页面ID:** atlasascendc_api_07_0970  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0970.html

---

#### 功能说明

算子编译部署后，会自动生成单算子API(aclnnxxx)接口，接口中的输入输出参数和算子原型定义中保持一致。

新增可选输入时，为了保持原有单算子API(aclnnxxx)接口的兼容性，可以通过Version接口配置aclnn接口的版本号，版本号需要从1开始配，且应该连续配置（和可选属性统一编号）。配置后，自动生成的aclnn接口会携带版本号。高版本号的接口会包含低版本号接口的所有参数。如下样例所示的原型定义：

```
class AddCustom : public OpDef {
   public:
    explicit AddCustom(const char* name) : OpDef(name) {
        this->Input("x")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("x1")
            .ParamType(OPTIONAL)
            .Version(1)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("x2")
            .ParamType(OPTIONAL)
            .Version(2)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->AICore().AddConfig("xxx");
    }
};
OP_ADD(AddCustom);
```

会自动生成3个版本的aclnn接口，定义如下：

```
aclnnStatus aclnnAddCustomGetWorkspaceSize(
    const aclTensorList *x,
    const aclTensorList *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);
aclnnStatus aclnnAddCustom(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    const aclrtStream stream);

aclnnStatus aclnnAddCustomV1GetWorkspaceSize(
    const aclTensorList *x,
    const aclTensor *x1Optional,
    const aclTensorList *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);
aclnnStatus aclnnAddCustomV1(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    const aclrtStream stream);

aclnnStatus aclnnAddCustomV2GetWorkspaceSize(
    const aclTensorList *x,
    const aclTensor *x1Optional,
    const aclTensor *x2Optional,
    const aclTensorList *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);
aclnnStatus aclnnAddCustomV2(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    const aclrtStream stream);
```

#### 函数原型

```
OpParamDef &Version(uint32_t version)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| **version** | 输入 | 指定的版本号。 |

#### 返回值说明

OpParamDef算子定义，OpParamDef请参考OpParamDef。

#### 约束说明

无
