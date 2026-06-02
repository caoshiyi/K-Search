# OpAICoreConfig注册接口（REGISTER\_OP\_AICORE\_CONFIG）<a name="ZH-CN_TOPIC_0000002523303608"></a>

## 功能说明<a name="zh-cn_topic_0000001600307121_zh-cn_topic_0000001576870453_zh-cn_topic_0000001525424352_section36583473819"></a>

不同的硬件形态算子原型定义不同的情况，可以通过新增[OpAICoreConfig](OpAICoreConfig.md)的方式，针对不同的AI处理器型号[注册差异化的算子原型](算子原型定义.md#section25861074132)。REGISTER\_OP\_AICORE\_CONFIG宏在不改变原有注册的基础上，允许单独新增文件来注册算子在不同硬件形态上的差异化信息。

使用该注册宏需要包含以下头文件：

```
#include "register/op_config_registry.h"
```

## 函数原型<a name="zh-cn_topic_0000001600307121_zh-cn_topic_0000001576870453_zh-cn_topic_0000001525424352_section13230182415108"></a>

```
REGISTER_OP_AICORE_CONFIG(opType, socVersion, opFunc)
```

## 参数说明<a name="zh-cn_topic_0000001600307121_zh-cn_topic_0000001576870453_zh-cn_topic_0000001525424352_section75395119104"></a>

<a name="zh-cn_topic_0000001600307121_zh-cn_topic_0000001576870453_zh-cn_topic_0000001575944081_table111938719446"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001600307121_zh-cn_topic_0000001576870453_zh-cn_topic_0000001575944081_row6223476444"><th class="cellrowborder" valign="top" width="17.22%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001600307121_zh-cn_topic_0000001576870453_zh-cn_topic_0000001575944081_p10223674448"><a name="zh-cn_topic_0000001600307121_zh-cn_topic_0000001576870453_zh-cn_topic_0000001575944081_p10223674448"></a><a name="zh-cn_topic_0000001600307121_zh-cn_topic_0000001576870453_zh-cn_topic_0000001575944081_p10223674448"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="15.340000000000002%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001600307121_zh-cn_topic_0000001576870453_zh-cn_topic_0000001575944081_p645511218169"><a name="zh-cn_topic_0000001600307121_zh-cn_topic_0000001576870453_zh-cn_topic_0000001575944081_p645511218169"></a><a name="zh-cn_topic_0000001600307121_zh-cn_topic_0000001576870453_zh-cn_topic_0000001575944081_p645511218169"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="67.44%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001600307121_zh-cn_topic_0000001576870453_zh-cn_topic_0000001575944081_p1922337124411"><a name="zh-cn_topic_0000001600307121_zh-cn_topic_0000001576870453_zh-cn_topic_0000001575944081_p1922337124411"></a><a name="zh-cn_topic_0000001600307121_zh-cn_topic_0000001576870453_zh-cn_topic_0000001575944081_p1922337124411"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001600307121_zh-cn_topic_0000001576870453_zh-cn_topic_0000001575944081_row152234713443"><td class="cellrowborder" valign="top" width="17.22%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001600307121_zh-cn_topic_0000001576870453_zh-cn_topic_0000001575944081_p318615392613"><a name="zh-cn_topic_0000001600307121_zh-cn_topic_0000001576870453_zh-cn_topic_0000001575944081_p318615392613"></a><a name="zh-cn_topic_0000001600307121_zh-cn_topic_0000001576870453_zh-cn_topic_0000001575944081_p318615392613"></a>opType</p>
</td>
<td class="cellrowborder" valign="top" width="15.340000000000002%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001600307121_zh-cn_topic_0000001576870453_zh-cn_topic_0000001575944081_p320343694214"><a name="zh-cn_topic_0000001600307121_zh-cn_topic_0000001576870453_zh-cn_topic_0000001575944081_p320343694214"></a><a name="zh-cn_topic_0000001600307121_zh-cn_topic_0000001576870453_zh-cn_topic_0000001575944081_p320343694214"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.44%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001600307121_zh-cn_topic_0000001576870453_p096733515614"><a name="zh-cn_topic_0000001600307121_zh-cn_topic_0000001576870453_p096733515614"></a><a name="zh-cn_topic_0000001600307121_zh-cn_topic_0000001576870453_p096733515614"></a>算子类型。</p>
</td>
</tr>
<tr id="row33032551664"><td class="cellrowborder" valign="top" width="17.22%" headers="mcps1.1.4.1.1 "><p id="p830415551762"><a name="p830415551762"></a><a name="p830415551762"></a>socVersion</p>
</td>
<td class="cellrowborder" valign="top" width="15.340000000000002%" headers="mcps1.1.4.1.2 "><p id="p103041655864"><a name="p103041655864"></a><a name="p103041655864"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.44%" headers="mcps1.1.4.1.3 "><p id="p1030417553617"><a name="p1030417553617"></a><a name="p1030417553617"></a>支持的AI处理器型号。</p>
</td>
</tr>
<tr id="row418410573612"><td class="cellrowborder" valign="top" width="17.22%" headers="mcps1.1.4.1.1 "><p id="p141847571966"><a name="p141847571966"></a><a name="p141847571966"></a>opFunc</p>
</td>
<td class="cellrowborder" valign="top" width="15.340000000000002%" headers="mcps1.1.4.1.2 "><p id="p31849571962"><a name="p31849571962"></a><a name="p31849571962"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.44%" headers="mcps1.1.4.1.3 "><p id="p61841157266"><a name="p61841157266"></a><a name="p61841157266"></a>返回OpAICoreConfig的回调函数指针，回调函数原型定义为：</p>
<a name="screen35771245210"></a><a name="screen35771245210"></a><pre class="screen" codetype="Cpp" id="screen35771245210">OpAICoreConfig (*)()</pre>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0000001600307121_zh-cn_topic_0000001576870453_zh-cn_topic_0000001525424352_section25791320141317"></a>

无

## 约束说明<a name="zh-cn_topic_0000001600307121_zh-cn_topic_0000001576870453_zh-cn_topic_0000001525424352_section19165124931511"></a>

若算子同时使用[AddConfig](AddConfig.md)注册算子支持的AI处理器型号以及OpAICoreConfig信息，且AI处理器型号相同时，通过AddConfig方式注册的配置优先级更高，会覆盖REGISTER\_OP\_AICORE\_CONFIG宏注册的OpAICoreConfig信息。

## 调用示例<a name="zh-cn_topic_0000001600307121_zh-cn_topic_0000001576870453_zh-cn_topic_0000001575944081_section320753512363"></a>

假设，已有原型注册文件op\_host/add\_custom.cpp实现如下，配置了算子支持的AI处理器型号ascendxxx1及算子输入输出原型信息：

```
...
namespace ops {
class AddCustom : public OpDef {
public:
    AddCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("z")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->AICore()
            .SetTiling(optiling::TilingFunc);
        // 请替换为实际的AI处理器型号
        this->AICore().AddConfig("ascendxxx1");
    }
};
OP_ADD(AddCustom);
} // namespace ops
```

可新增文件op\_host/add\_custom\_xxx.cpp，使用REGISTER\_OP\_AICORE\_CONFIG单独注册算子支持的AI处理器型号ascendxxx2，示例如下：

```
#include "register/op_config_registry.h"
namespace ops {
REGISTER_OP_AICORE_CONFIG(AddCustom, ascendxxx2, []() {
    ops::OpAICoreConfig config("ascendxxx2");
    return config;
});
}
```

