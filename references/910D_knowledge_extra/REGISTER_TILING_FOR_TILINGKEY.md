# REGISTER\_TILING\_FOR\_TILINGKEY<a name="ZH-CN_TOPIC_0000002554344493"></a>

## 产品支持情况<a name="section1550532418810"></a>

<a name="table38301303189"></a>
<table><thead align="left"><tr id="row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p1883113061818"><a name="p1883113061818"></a><a name="p1883113061818"></a><span id="ph20833205312295"><a name="ph20833205312295"></a><a name="ph20833205312295"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p783113012187"><a name="p783113012187"></a><a name="p783113012187"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row1272474920205"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p17301775812"><a name="p17301775812"></a><a name="p17301775812"></a><span id="ph2272194216543"><a name="ph2272194216543"></a><a name="ph2272194216543"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p37256491200"><a name="p37256491200"></a><a name="p37256491200"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="zh-cn_topic_0000001526206862_section212607105720"></a>

用于在kernel侧注册与TilingKey相匹配的TilingData自定义结构体；该接口需提供一个逻辑表达式，逻辑表达式以字符串“TILING\_KEY\_VAR”代指实际TilingKey，表达TilingKey所满足的范围。

## 函数原型<a name="zh-cn_topic_0000001526206862_section1630753514297"></a>

```
REGISTER_TILING_FOR_TILINGKEY(EXPRESSION, TILING_STRUCT)
```

## 参数说明<a name="zh-cn_topic_0000001526206862_section129451113125413"></a>

<a name="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_table111938719446"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_row6223476444"><th class="cellrowborder" valign="top" width="17.22%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p10223674448"><a name="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p10223674448"></a><a name="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p10223674448"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="15.340000000000002%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p645511218169"><a name="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p645511218169"></a><a name="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p645511218169"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="67.44%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p1922337124411"><a name="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p1922337124411"></a><a name="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p1922337124411"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_row152234713443"><td class="cellrowborder" valign="top" width="17.22%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p2340183613156"><a name="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p2340183613156"></a><a name="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p2340183613156"></a>EXPRESSION</p>
</td>
<td class="cellrowborder" valign="top" width="15.340000000000002%" headers="mcps1.1.4.1.2 "><p id="p137163271873"><a name="p137163271873"></a><a name="p137163271873"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.44%" headers="mcps1.1.4.1.3 "><p id="p1190216251682"><a name="p1190216251682"></a><a name="p1190216251682"></a>EXPRESSION为逻辑运算，其中用TILING_KEY_VAR指代TilingKey。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001526206862_row1239183183016"><td class="cellrowborder" valign="top" width="17.22%" headers="mcps1.1.4.1.1 "><p id="p17557191817713"><a name="p17557191817713"></a><a name="p17557191817713"></a>TILING_STRUCT</p>
</td>
<td class="cellrowborder" valign="top" width="15.340000000000002%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001526206862_p7239938308"><a name="zh-cn_topic_0000001526206862_p7239938308"></a><a name="zh-cn_topic_0000001526206862_p7239938308"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.44%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001526206862_p72396320307"><a name="zh-cn_topic_0000001526206862_p72396320307"></a><a name="zh-cn_topic_0000001526206862_p72396320307"></a>用户注册的与TilingKey相匹配的TilingData自定义结构体。</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="zh-cn_topic_0000001526206862_section65498832"></a>

-   使用该接口时，需确保已使用REGISTER\_TILING\_DEFAULT注册默认的用户自定义TilingData结构体，用于告知框架侧用户使用标准C++语法来定义TilingData。
-   EXPRESSION当前支持位运算：&、|、\~、^；移位运算符：<<、\>\>；算术运算：+、-、\*、/、%；条件运算符：==、!=、\>、<、\>=、<=；逻辑与&&、或||以及\(\)。优先级同C++。
-   若TilingData结构体在命名空间内，注册时需要携带对应的命名空间作用域符。
-   不支持同个TilingKey指向不同TilingData结构体，会出现拦截报错。
-   暂不支持kernel直调工程。

## 调用示例<a name="zh-cn_topic_0000001526206862_section97001499599"></a>

```
extern "C" __global__ __aicore__ void add_custom(__gm__ uint8_t *x, __gm__ uint8_t *y, __gm__ uint8_t *z, __gm__ uint8_t *tiling)
{
    REGISTER_TILING_DEFAULT(optiling::TilingData);  // 注册用户默认自定义TilingData结构体
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 1", optiling::TilingDataA); // 注册TilingKey为1的TilingData结构体
    REGISTER_TILING_FOR_TILINGKEY("(TILING_KEY_VAR >= 10) && (TILING_KEY_VAR <= 15)", optiling::TilingDataB); // 注册TilingKey在[10,15]之间的TilingData结构体
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR & 0xFF", optiling::TilingDataC); // 注册TilingKey低16位为1的TilingData结构体
    if (TILING_KEY_IS(1)) {
        GET_TILING_DATA_WITH_STRUCT(optiling::TilingDataA, tilingData, tiling);
        ......
    } else if (TILING_KEY_IS(11)) {
        GET_TILING_DATA_WITH_STRUCT(optiling::TilingDataB, tilingData, tiling);
        ......
    } else if (TILING_KEY_IS(14)) {
        GET_TILING_DATA_WITH_STRUCT(optiling::TilingDataB, tilingData, tiling);
        ......
    } else if (TILING_KEY_IS(255)) {
        GET_TILING_DATA_WITH_STRUCT(optiling::TilingDataC, tilingData, tiling);
        ......
    } else {
        GET_TILING_DATA(tilingData, tiling);
        ......
    }
}
```

使用标准C++语法注册tiling结构体：

```
class TilingDataA{
public:
    ...
};
class TilingDataB{
public:
    ...
};
class TilingDataC{
public:
    ...
};
```

配套的host侧tiling函数示例：

```
ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    // 其他代码逻辑
    ...
    if(condition1){
        context->SetTilingKey(1);
        optiling::TilingDataA *Addtiling = context->GetTilingData<optiling::TilingDataA>();
        ...
    } else if (condition2){
        context->SetTilingKey(11);
        optiling::TilingDataB *Addtiling = context->GetTilingData<optiling::TilingDataB >();
        ...
    } else if (condition3){
        context->SetTilingKey(14);
        optiling::TilingDataB *Addtiling = context->GetTilingData<optiling::TilingDataB >();
        ...
    } else if (condition4){
        context->SetTilingKey(255);
        optiling::TilingDataC *Addtiling = context->GetTilingData<optiling::TilingDataC >();
        ...
    }
    ...
    // 其他代码逻辑
}
```

