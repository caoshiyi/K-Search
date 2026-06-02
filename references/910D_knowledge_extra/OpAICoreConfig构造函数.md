# OpAICoreConfig构造函数<a name="ZH-CN_TOPIC_0000002523304752"></a>

## 功能说明<a name="zh-cn_topic_0000002078486173_zh-cn_topic_0000001576727153_zh-cn_topic_0000001389787297_section36583473819"></a>

OpAICoreConfig构造函数。

## 函数原型<a name="zh-cn_topic_0000002078486173_zh-cn_topic_0000001576727153_zh-cn_topic_0000001389787297_section13230182415108"></a>

```
OpAICoreConfig()
OpAICoreConfig(const char *soc)
```

## 参数说明<a name="zh-cn_topic_0000002078486173_zh-cn_topic_0000001576727153_zh-cn_topic_0000001389787297_section75395119104"></a>

<a name="zh-cn_topic_0000002078486173_zh-cn_topic_0000001576727153_zh-cn_topic_0000001389787297_table111938719446"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000002078486173_zh-cn_topic_0000001576727153_zh-cn_topic_0000001389787297_row6223476444"><th class="cellrowborder" valign="top" width="17.22%" id="mcps1.1.4.1.1"><p id="p1085176175119"><a name="p1085176175119"></a><a name="p1085176175119"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="15.340000000000002%" id="mcps1.1.4.1.2"><p id="p1851763519"><a name="p1851763519"></a><a name="p1851763519"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="67.44%" id="mcps1.1.4.1.3"><p id="p148519610515"><a name="p148519610515"></a><a name="p148519610515"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000002078486173_zh-cn_topic_0000001576727153_zh-cn_topic_0000001389787297_row152234713443"><td class="cellrowborder" valign="top" width="17.22%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002078486173_zh-cn_topic_0000001576727153_zh-cn_topic_0000001389787297_p8563195616313"><a name="zh-cn_topic_0000002078486173_zh-cn_topic_0000001576727153_zh-cn_topic_0000001389787297_p8563195616313"></a><a name="zh-cn_topic_0000002078486173_zh-cn_topic_0000001576727153_zh-cn_topic_0000001389787297_p8563195616313"></a>soc</p>
</td>
<td class="cellrowborder" valign="top" width="15.340000000000002%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002078486173_zh-cn_topic_0000001576727153_zh-cn_topic_0000001389787297_p15663137127"><a name="zh-cn_topic_0000002078486173_zh-cn_topic_0000001576727153_zh-cn_topic_0000001389787297_p15663137127"></a><a name="zh-cn_topic_0000002078486173_zh-cn_topic_0000001576727153_zh-cn_topic_0000001389787297_p15663137127"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.44%" headers="mcps1.1.4.1.3 "><p id="p823866165711"><a name="p823866165711"></a><a name="p823866165711"></a>AI处理器型号。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0000002078486173_zh-cn_topic_0000001576727153_zh-cn_topic_0000001389787297_section25791320141317"></a>

无

## 约束说明<a name="zh-cn_topic_0000002078486173_zh-cn_topic_0000001576727153_zh-cn_topic_0000001389787297_section19165124931511"></a>

传入soc入参的构造函数会对OpAICoreConfig结构中的部分参数进行初始化，具体的参数和初始化值如下表所示：

<a name="table56008147710"></a>
<table><thead align="left"><tr id="row3601014470"><th class="cellrowborder" valign="top" width="18.09%" id="mcps1.1.4.1.1"><p id="p19601111411713"><a name="p19601111411713"></a><a name="p19601111411713"></a>配置参数</p>
</th>
<th class="cellrowborder" valign="top" width="60.589999999999996%" id="mcps1.1.4.1.2"><p id="p14573171417333"><a name="p14573171417333"></a><a name="p14573171417333"></a>说明</p>
</th>
<th class="cellrowborder" valign="top" width="21.32%" id="mcps1.1.4.1.3"><p id="p56011114871"><a name="p56011114871"></a><a name="p56011114871"></a>初始化值</p>
</th>
</tr>
</thead>
<tbody><tr id="row460191414720"><td class="cellrowborder" valign="top" width="18.09%" headers="mcps1.1.4.1.1 "><p id="p136012142712"><a name="p136012142712"></a><a name="p136012142712"></a><a href="DynamicCompileStaticFlag.md">DynamicCompileStaticFlag</a></p>
</td>
<td class="cellrowborder" valign="top" width="60.589999999999996%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001575931912_p183604264191"><a name="zh-cn_topic_0000001575931912_p183604264191"></a><a name="zh-cn_topic_0000001575931912_p183604264191"></a>用于标识该算子实现是否支持入图时的静态Shape编译。</p>
</td>
<td class="cellrowborder" valign="top" width="21.32%" headers="mcps1.1.4.1.3 "><p id="p15601114777"><a name="p15601114777"></a><a name="p15601114777"></a>true</p>
</td>
</tr>
<tr id="row2060161415717"><td class="cellrowborder" valign="top" width="18.09%" headers="mcps1.1.4.1.1 "><p id="p26019142078"><a name="p26019142078"></a><a name="p26019142078"></a><a href="DynamicFormatFlag.md">DynamicFormatFlag</a></p>
</td>
<td class="cellrowborder" valign="top" width="60.589999999999996%" headers="mcps1.1.4.1.2 "><p id="p145734149339"><a name="p145734149339"></a><a name="p145734149339"></a>标识是否根据<a href="SetOpSelectFormat.md">SetOpSelectFormat</a>设置的函数自动推导算子输入输出支持的dtype和format。</p>
</td>
<td class="cellrowborder" valign="top" width="21.32%" headers="mcps1.1.4.1.3 "><p id="p56015149714"><a name="p56015149714"></a><a name="p56015149714"></a>true</p>
</td>
</tr>
<tr id="row9601014375"><td class="cellrowborder" valign="top" width="18.09%" headers="mcps1.1.4.1.1 "><p id="p116011714075"><a name="p116011714075"></a><a name="p116011714075"></a><a href="DynamicRankSupportFlag.md">DynamicRankSupportFlag</a></p>
</td>
<td class="cellrowborder" valign="top" width="60.589999999999996%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001575612432_zh-cn_topic_0000001576875005_zh-cn_topic_0000001525424352_p10448657201310"><a name="zh-cn_topic_0000001575612432_zh-cn_topic_0000001576875005_zh-cn_topic_0000001525424352_p10448657201310"></a><a name="zh-cn_topic_0000001575612432_zh-cn_topic_0000001576875005_zh-cn_topic_0000001525424352_p10448657201310"></a>标识算子是否支持dynamicRank（动态维度）。</p>
</td>
<td class="cellrowborder" valign="top" width="21.32%" headers="mcps1.1.4.1.3 "><p id="p66017146713"><a name="p66017146713"></a><a name="p66017146713"></a>true</p>
</td>
</tr>
<tr id="row126011142077"><td class="cellrowborder" valign="top" width="18.09%" headers="mcps1.1.4.1.1 "><p id="p2601101420710"><a name="p2601101420710"></a><a name="p2601101420710"></a><a href="DynamicShapeSupportFlag.md">DynamicShapeSupportFlag</a></p>
</td>
<td class="cellrowborder" valign="top" width="60.589999999999996%" headers="mcps1.1.4.1.2 "><p id="p5890739574"><a name="p5890739574"></a><a name="p5890739574"></a>用于标识该算子是否支持入图时的动态Shape场景。</p>
</td>
<td class="cellrowborder" valign="top" width="21.32%" headers="mcps1.1.4.1.3 "><p id="p1160131416714"><a name="p1160131416714"></a><a name="p1160131416714"></a>true</p>
</td>
</tr>
<tr id="row160116140712"><td class="cellrowborder" valign="top" width="18.09%" headers="mcps1.1.4.1.1 "><p id="p16601914377"><a name="p16601914377"></a><a name="p16601914377"></a><a href="NeedCheckSupportFlag.md">NeedCheckSupportFlag</a></p>
</td>
<td class="cellrowborder" valign="top" width="60.589999999999996%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001626211657_p1114624162912"><a name="zh-cn_topic_0000001626211657_p1114624162912"></a><a name="zh-cn_topic_0000001626211657_p1114624162912"></a>标识是否在算子融合阶段调用算子参数校验函数进行data type与Shape的校验。</p>
</td>
<td class="cellrowborder" valign="top" width="21.32%" headers="mcps1.1.4.1.3 "><p id="p1360141410720"><a name="p1360141410720"></a><a name="p1360141410720"></a>false</p>
</td>
</tr>
<tr id="row5601181415716"><td class="cellrowborder" valign="top" width="18.09%" headers="mcps1.1.4.1.1 "><p id="p13601181417712"><a name="p13601181417712"></a><a name="p13601181417712"></a><a href="PrecisionReduceFlag.md">PrecisionReduceFlag</a></p>
</td>
<td class="cellrowborder" valign="top" width="60.589999999999996%" headers="mcps1.1.4.1.2 "><p id="p957351413338"><a name="p957351413338"></a><a name="p957351413338"></a>此字段用于进行ATC模型转换或者进行网络调测时，控制算子的精度模式。</p>
</td>
<td class="cellrowborder" valign="top" width="21.32%" headers="mcps1.1.4.1.3 "><p id="p46013144712"><a name="p46013144712"></a><a name="p46013144712"></a>true</p>
</td>
</tr>
</tbody>
</table>

无入参的默认构造函数不会初始化上述参数。

## 调用示例<a name="zh-cn_topic_0000002078486173_zh-cn_topic_0000001576727153_zh-cn_topic_0000001389787297_section320753512363"></a>

```
class AddCustom : public OpDef {
public:
    AddCustom(const char* name) : OpDef(name)
    {
        this->Input("x").DataType({ ge::DT_FLOAT16 }).ParamType(OPTIONAL);
        this->Output("y").DataType({ ge::DT_FLOAT16 });
        // 使用soc入参构造函数
        OpAICoreConfig aicConfig1("ascendxxx1");
        OpAICoreConfig aicConfig2("ascendxxx2");
        aicConfig1.Input("x")
            .ParamType(OPTIONAL)
            .DataType({ ge::DT_FLOAT })
            .Format({ ge::FORMAT_ND });
        aicConfig2.Input("x")
            .ParamType(REQUIRED)
            .DataType({ ge::DT_INT32 })
            .Format({ ge::FORMAT_ND });
        this->AICore().AddConfig("ascendxxx1", aicConfig1);
        this->AICore().AddConfig("ascendxxx2", aicConfig2);
    }
};
```

