# HcclServerType<a name="ZH-CN_TOPIC_0000002554424793"></a>

## 功能说明<a name="zh-cn_topic_0000001962490173_zh-cn_topic_0000001339105488_section36583473819"></a>

配置HCCL的服务端类型。

## 函数原型<a name="zh-cn_topic_0000001962490173_zh-cn_topic_0000001339105488_section13230182415108"></a>

```
void HcclServerType(enum HcclServerType type, const char *soc=nullptr)
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
<tbody><tr id="zh-cn_topic_0000001962490173_zh-cn_topic_0000001339105488_row152234713443"><td class="cellrowborder" valign="top" width="17.22%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001962490173_zh-cn_topic_0000001339105488_p2340183613156"><a name="zh-cn_topic_0000001962490173_zh-cn_topic_0000001339105488_p2340183613156"></a><a name="zh-cn_topic_0000001962490173_zh-cn_topic_0000001339105488_p2340183613156"></a>type</p>
</td>
<td class="cellrowborder" valign="top" width="15.340000000000002%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001962490173_zh-cn_topic_0000001339105488_p320343694214"><a name="zh-cn_topic_0000001962490173_zh-cn_topic_0000001339105488_p320343694214"></a><a name="zh-cn_topic_0000001962490173_zh-cn_topic_0000001339105488_p320343694214"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.44%" headers="mcps1.1.4.1.3 "><p id="p895664212548"><a name="p895664212548"></a><a name="p895664212548"></a>HCCL的服务端类型，类型为HcclServerType枚举类，定义如下：</p>
<a name="screen112171935175919"></a><a name="screen112171935175919"></a><pre class="screen" codetype="Cpp" id="screen112171935175919">namespace ops{
enum HcclServerType : uint32_t {
    AICPU = 0,  // AI CPU服务端
    AICORE, // AI Core服务端
    CCU,    // CCU服务端，仅在<span id="ph18742853192816"><a name="ph18742853192816"></a><a name="ph18742853192816"></a>AI处理器</span>包含CCU单元时支持
    MAX     // 预留参数，不支持使用
};
}</pre>
</td>
</tr>
<tr id="row122371621145118"><td class="cellrowborder" valign="top" width="17.22%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002078486173_zh-cn_topic_0000001576727153_zh-cn_topic_0000001389787297_p8563195616313"><a name="zh-cn_topic_0000002078486173_zh-cn_topic_0000001576727153_zh-cn_topic_0000001389787297_p8563195616313"></a><a name="zh-cn_topic_0000002078486173_zh-cn_topic_0000001576727153_zh-cn_topic_0000001389787297_p8563195616313"></a>soc</p>
</td>
<td class="cellrowborder" valign="top" width="15.340000000000002%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002078486173_zh-cn_topic_0000001576727153_zh-cn_topic_0000001389787297_p15663137127"><a name="zh-cn_topic_0000002078486173_zh-cn_topic_0000001576727153_zh-cn_topic_0000001389787297_p15663137127"></a><a name="zh-cn_topic_0000002078486173_zh-cn_topic_0000001576727153_zh-cn_topic_0000001389787297_p15663137127"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.44%" headers="mcps1.1.4.1.3 "><p id="p991524490"><a name="p991524490"></a><a name="p991524490"></a><span id="ph19321164217255"><a name="ph19321164217255"></a><a name="ph19321164217255"></a>AI处理器</span>型号。为该型号配置服务端类型。</p>
<p id="p366542912619"><a name="p366542912619"></a><a name="p366542912619"></a>可选参数，nullptr或者""表示为算子支持的所有型号配置服务端类型。</p>
<p id="p94761732201220"><a name="p94761732201220"></a><a name="p94761732201220"></a>soc取值需确保在算子支持的<span id="ph3848185012164"><a name="ph3848185012164"></a><a name="ph3848185012164"></a>AI处理器</span>型号范围内，即已经调用<a href="AddConfig.md">AddConfig</a>接口注册。</p>
<p id="zh-cn_topic_0000001575929572_p14611945131415"><a name="zh-cn_topic_0000001575929572_p14611945131415"></a><a name="zh-cn_topic_0000001575929572_p14611945131415"></a>填写规则请参考<span>算子工程目录下编译配置项文件CMakePresets.json中的ASCEND_COMPUTE_UNIT字段，该字段取值在</span><a href="创建算子工程.md">使用msOpGen创建工程</a><span>时自动生成</span>。</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="zh-cn_topic_0000001962490173_zh-cn_topic_0000001339105488_section19165124931511"></a>

-   使用该接口前，算子需要先通过[MC2](MC2.md)接口注册该算子是通算融合算子，注册后即返回一个[OpMC2Def](OpMC2Def.md)结构。
-   同时为特定AI处理器型号和所有AI处理器型号配置服务端类型时，特定AI处理器型号配置的优先级更高。

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
        this->AICore().AddConfig("ascendxxx1");
        this->AICore().AddConfig("ascendxxx2");
        this->MC2().HcclGroup("group"); // 配置通信域名称为group
        this->MC2().HcclServerType(HcclServerType::AICPU, "ascendxxx1"); // 配置ascendxxx1型号的通信模式为AI CPU
        this->MC2().HcclServerType(HcclServerType::AICORE); // 配置其他型号即ascendxxx2的通信模式为AI Core
    }
};
OP_ADD(MC2Custom);
```

