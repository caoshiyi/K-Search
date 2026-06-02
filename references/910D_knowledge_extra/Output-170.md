# Output<a name="ZH-CN_TOPIC_0000002554344297"></a>

## 功能说明<a name="zh-cn_topic_0000001760150049_zh-cn_topic_0000001576870901_zh-cn_topic_0000001575944081_section36583473819"></a>

某些场景下，同一个算子在不同的AI处理器型号上，其支持的原型输出不同。

通过该接口，可针对不同的AI处理器型号注册差异化的算子输出。调用该接口后会返回一个OpParamDef结构，后续可通过该结构配置算子输出信息。

## 函数原型<a name="zh-cn_topic_0000001760150049_zh-cn_topic_0000001576870901_zh-cn_topic_0000001575944081_section13230182415108"></a>

```
OpParamDef &Output(const char *name)
```

## 参数说明<a name="zh-cn_topic_0000001760150049_zh-cn_topic_0000001576870901_zh-cn_topic_0000001575944081_section75395119104"></a>

<a name="zh-cn_topic_0000001760150049_zh-cn_topic_0000001576870901_zh-cn_topic_0000001575944081_table111938719446"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001760150049_zh-cn_topic_0000001576870901_zh-cn_topic_0000001575944081_row6223476444"><th class="cellrowborder" valign="top" width="17.22%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001760150049_zh-cn_topic_0000001576870901_zh-cn_topic_0000001575944081_p10223674448"><a name="zh-cn_topic_0000001760150049_zh-cn_topic_0000001576870901_zh-cn_topic_0000001575944081_p10223674448"></a><a name="zh-cn_topic_0000001760150049_zh-cn_topic_0000001576870901_zh-cn_topic_0000001575944081_p10223674448"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="15.340000000000002%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001760150049_zh-cn_topic_0000001576870901_zh-cn_topic_0000001575944081_p645511218169"><a name="zh-cn_topic_0000001760150049_zh-cn_topic_0000001576870901_zh-cn_topic_0000001575944081_p645511218169"></a><a name="zh-cn_topic_0000001760150049_zh-cn_topic_0000001576870901_zh-cn_topic_0000001575944081_p645511218169"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="67.44%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001760150049_zh-cn_topic_0000001576870901_zh-cn_topic_0000001575944081_p1922337124411"><a name="zh-cn_topic_0000001760150049_zh-cn_topic_0000001576870901_zh-cn_topic_0000001575944081_p1922337124411"></a><a name="zh-cn_topic_0000001760150049_zh-cn_topic_0000001576870901_zh-cn_topic_0000001575944081_p1922337124411"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001760150049_zh-cn_topic_0000001576870901_zh-cn_topic_0000001575944081_row152234713443"><td class="cellrowborder" valign="top" width="17.22%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001760150049_zh-cn_topic_0000001576870901_zh-cn_topic_0000001575944081_p318615392613"><a name="zh-cn_topic_0000001760150049_zh-cn_topic_0000001576870901_zh-cn_topic_0000001575944081_p318615392613"></a><a name="zh-cn_topic_0000001760150049_zh-cn_topic_0000001576870901_zh-cn_topic_0000001575944081_p318615392613"></a>name</p>
</td>
<td class="cellrowborder" valign="top" width="15.340000000000002%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001760150049_zh-cn_topic_0000001576870901_zh-cn_topic_0000001575944081_p320343694214"><a name="zh-cn_topic_0000001760150049_zh-cn_topic_0000001576870901_zh-cn_topic_0000001575944081_p320343694214"></a><a name="zh-cn_topic_0000001760150049_zh-cn_topic_0000001576870901_zh-cn_topic_0000001575944081_p320343694214"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.44%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001760150049_zh-cn_topic_0000001576870901_zh-cn_topic_0000001575944081_p453018873120"><a name="zh-cn_topic_0000001760150049_zh-cn_topic_0000001576870901_zh-cn_topic_0000001575944081_p453018873120"></a><a name="zh-cn_topic_0000001760150049_zh-cn_topic_0000001576870901_zh-cn_topic_0000001575944081_p453018873120"></a>算子输出名称。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0000001760150049_zh-cn_topic_0000001576870901_zh-cn_topic_0000001575944081_section25791320141317"></a>

算子参数定义，OpParamDef请参考[OpParamDef](OpParamDef.md)。

## 约束说明<a name="zh-cn_topic_0000001760150049_zh-cn_topic_0000001576870901_zh-cn_topic_0000001575944081_section19165124931511"></a>

无

## 调用示例<a name="zh-cn_topic_0000001760150049_zh-cn_topic_0000001576870901_zh-cn_topic_0000001575944081_section320753512363"></a>

```
class AddCustom : public OpDef {
public:
    AddCustom(const char* name) : OpDef(name)
    {
        this->Input("x").DataType({ ge::DT_FLOAT16 }).ParamType(OPTIONAL);
        this->Output("y").DataType({ ge::DT_FLOAT16 });
        OpAICoreConfig aicConfig1;
        OpAICoreConfig aicConfig2;
        aicConfig1.Output("y")
            .ParamType(OPTIONAL)
            .DataType({ ge::DT_FLOAT })
            .Format({ ge::FORMAT_ND });
        aicConfig2.Output("y")
            .ParamType(REQUIRED)
            .DataType({ ge::DT_INT32 })
            .Format({ ge::FORMAT_ND });
        this->AICore().AddConfig("ascendxxx1", aicConfig1);
        this->AICore().AddConfig("ascendxxx2", aicConfig2);
    }
};
```

