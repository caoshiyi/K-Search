# HcclGroup<a name="ZH-CN_TOPIC_0000002554344321"></a>

## 功能说明<a name="zh-cn_topic_0000001962490173_zh-cn_topic_0000001339105488_section36583473819"></a>

配置通信域名称，每个名称对应一个通信域。配置后在Kernel侧调用GetHcclContext接口可获取通信域对应的context（消息区）地址。

## 函数原型<a name="zh-cn_topic_0000001962490173_zh-cn_topic_0000001339105488_section13230182415108"></a>

```
OpMC2Def &HcclGroup(const char *value)
OpMC2Def &HcclGroup(std::vector<const char *> value)
```

## 参数说明<a name="zh-cn_topic_0000001962490173_zh-cn_topic_0000001339105488_section75395119104"></a>

<a name="zh-cn_topic_0000001962490173_zh-cn_topic_0000001339105488_table111938719446"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001962490173_zh-cn_topic_0000001339105488_row6223476444"><th class="cellrowborder" valign="top" width="17.22%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001962490173_zh-cn_topic_0000001339105488_p10223674448"><a name="zh-cn_topic_0000001962490173_zh-cn_topic_0000001339105488_p10223674448"></a><a name="zh-cn_topic_0000001962490173_zh-cn_topic_0000001339105488_p10223674448"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="15.340000000000002%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001962490173_zh-cn_topic_0000001339105488_p645511218169"><a name="zh-cn_topic_0000001962490173_zh-cn_topic_0000001339105488_p645511218169"></a><a name="zh-cn_topic_0000001962490173_zh-cn_topic_0000001339105488_p645511218169"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="67.44%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001962490173_zh-cn_topic_0000001339105488_p1922337124411"><a name="zh-cn_topic_0000001962490173_zh-cn_topic_0000001339105488_p1922337124411"></a><a name="zh-cn_topic_0000001962490173_zh-cn_topic_0000001339105488_p1922337124411"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001962490173_zh-cn_topic_0000001339105488_row152234713443"><td class="cellrowborder" valign="top" width="17.22%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001962490173_zh-cn_topic_0000001339105488_p2340183613156"><a name="zh-cn_topic_0000001962490173_zh-cn_topic_0000001339105488_p2340183613156"></a><a name="zh-cn_topic_0000001962490173_zh-cn_topic_0000001339105488_p2340183613156"></a>value</p>
</td>
<td class="cellrowborder" valign="top" width="15.340000000000002%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001962490173_zh-cn_topic_0000001339105488_p320343694214"><a name="zh-cn_topic_0000001962490173_zh-cn_topic_0000001339105488_p320343694214"></a><a name="zh-cn_topic_0000001962490173_zh-cn_topic_0000001339105488_p320343694214"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.44%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001962490173_zh-cn_topic_0000001339105488_p2684123934216"><a name="zh-cn_topic_0000001962490173_zh-cn_topic_0000001339105488_p2684123934216"></a><a name="zh-cn_topic_0000001962490173_zh-cn_topic_0000001339105488_p2684123934216"></a>配置的通信域名称。单个通信域使用const char *，多通信域使用std::vector&lt;const char *&gt;。</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="zh-cn_topic_0000001962490173_zh-cn_topic_0000001339105488_section19165124931511"></a>

使用该接口前，算子需要先通过[MC2](MC2.md)接口注册该算子是通算融合算子，注册后即返回一个[OpMC2Def](OpMC2Def.md)结构。

通信域名称必须先配置为REQUIRED String类型的属性，属性名即为通信域名称。

## 调用示例<a name="zh-cn_topic_0000001962490173_section163549032418"></a>

```
class MC2Custom : public OpDef {
public:
    MC2Custom(const char* name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("z").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Attr("group").AttrType(REQUIRED).String();
        this->AICore().AddConfig("ascendxxx");
        this->MC2().HcclGroup("group"); // 配置通信域名称为group
    }
};
OP_ADD(MC2Custom);
```

