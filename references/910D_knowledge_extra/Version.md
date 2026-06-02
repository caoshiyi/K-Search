# Version<a name="ZH-CN_TOPIC_0000002523304250"></a>

## 功能说明<a name="zh-cn_topic_0000001797014949_zh-cn_topic_0000001526594958_zh-cn_topic_0000001525424352_section36583473819"></a>

算子编译部署后，会自动生成单算子API\(aclnnxxx\)接口，接口中的输入输出参数和算子原型定义中保持一致。

新增可选输入时，为了保持原有单算子API\(aclnnxxx\)接口的兼容性，可以通过Version接口配置aclnn接口的版本号，版本号需要从1开始配，且应该连续配置（和[可选属性](OpAttrDef-167.md)统一编号）。配置后，自动生成的aclnn接口会携带版本号。高版本号的接口会包含低版本号接口的所有参数。如下样例所示的原型定义：

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

## 函数原型<a name="zh-cn_topic_0000001797014949_zh-cn_topic_0000001526594958_zh-cn_topic_0000001525424352_section13230182415108"></a>

```
OpParamDef &Version(uint32_t version)
```

## 参数说明<a name="zh-cn_topic_0000001797014949_zh-cn_topic_0000001526594958_zh-cn_topic_0000001525424352_section75395119104"></a>

<a name="zh-cn_topic_0000001797014949_zh-cn_topic_0000001526594958_zh-cn_topic_0000001575944081_table111938719446"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001797014949_zh-cn_topic_0000001526594958_zh-cn_topic_0000001575944081_row6223476444"><th class="cellrowborder" valign="top" width="17.22%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001797014949_zh-cn_topic_0000001526594958_zh-cn_topic_0000001575944081_p10223674448"><a name="zh-cn_topic_0000001797014949_zh-cn_topic_0000001526594958_zh-cn_topic_0000001575944081_p10223674448"></a><a name="zh-cn_topic_0000001797014949_zh-cn_topic_0000001526594958_zh-cn_topic_0000001575944081_p10223674448"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="15.340000000000002%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001797014949_zh-cn_topic_0000001526594958_zh-cn_topic_0000001575944081_p645511218169"><a name="zh-cn_topic_0000001797014949_zh-cn_topic_0000001526594958_zh-cn_topic_0000001575944081_p645511218169"></a><a name="zh-cn_topic_0000001797014949_zh-cn_topic_0000001526594958_zh-cn_topic_0000001575944081_p645511218169"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="67.44%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001797014949_zh-cn_topic_0000001526594958_zh-cn_topic_0000001575944081_p1922337124411"><a name="zh-cn_topic_0000001797014949_zh-cn_topic_0000001526594958_zh-cn_topic_0000001575944081_p1922337124411"></a><a name="zh-cn_topic_0000001797014949_zh-cn_topic_0000001526594958_zh-cn_topic_0000001575944081_p1922337124411"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001797014949_zh-cn_topic_0000001526594958_zh-cn_topic_0000001575944081_row152234713443"><td class="cellrowborder" valign="top" width="17.22%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001797014949_zh-cn_topic_0000001526594958_zh-cn_topic_0000001575944081_p318615392613"><a name="zh-cn_topic_0000001797014949_zh-cn_topic_0000001526594958_zh-cn_topic_0000001575944081_p318615392613"></a><a name="zh-cn_topic_0000001797014949_zh-cn_topic_0000001526594958_zh-cn_topic_0000001575944081_p318615392613"></a><strong id="zh-cn_topic_0000001797014949_b97895303465"><a name="zh-cn_topic_0000001797014949_b97895303465"></a><a name="zh-cn_topic_0000001797014949_b97895303465"></a>version</strong></p>
</td>
<td class="cellrowborder" valign="top" width="15.340000000000002%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001797014949_zh-cn_topic_0000001526594958_zh-cn_topic_0000001575944081_p320343694214"><a name="zh-cn_topic_0000001797014949_zh-cn_topic_0000001526594958_zh-cn_topic_0000001575944081_p320343694214"></a><a name="zh-cn_topic_0000001797014949_zh-cn_topic_0000001526594958_zh-cn_topic_0000001575944081_p320343694214"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.44%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001797014949_zh-cn_topic_0000001526594958_p096733515614"><a name="zh-cn_topic_0000001797014949_zh-cn_topic_0000001526594958_p096733515614"></a><a name="zh-cn_topic_0000001797014949_zh-cn_topic_0000001526594958_p096733515614"></a>指定的版本号。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0000001797014949_zh-cn_topic_0000001526594958_zh-cn_topic_0000001525424352_section25791320141317"></a>

OpParamDef算子定义，OpParamDef请参考[OpParamDef](OpParamDef.md)。

## 约束说明<a name="zh-cn_topic_0000001797014949_zh-cn_topic_0000001526594958_zh-cn_topic_0000001525424352_section19165124931511"></a>

无

