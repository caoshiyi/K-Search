# ASCENDC\_TPL\_SEL\_PARAM<a name="ZH-CN_TOPIC_0000002554423921"></a>

## 功能说明<a name="zh-cn_topic_0000001867289945_zh-cn_topic_0000001389787297_section36583473819"></a>

Tiling模板编程时，开发者通过调用此接口自动生成并配置TilingKey。

使用该接口需要包含定义模板参数和模板参数组合的头文件。详细内容请参考[Tiling模板编程](Tiling模板编程.md)。

## 函数原型<a name="zh-cn_topic_0000001867289945_zh-cn_topic_0000001389787297_section13230182415108"></a>

```
#define ASCENDC_TPL_SEL_PARAM(context, ...)           \
do {                                                  \
    uint64_t key = GET_TPL_TILING_KEY({__VA_ARGS__}); \
    context->SetTilingKey(key);                       \
} while(0)
// context指代TilingFunc(gert::TilingContext *context)中的context
```

## 参数说明<a name="zh-cn_topic_0000001867289945_zh-cn_topic_0000001389787297_section75395119104"></a>

<a name="zh-cn_topic_0000001820490272_zh-cn_topic_0000001389787297_table111938719446"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001820490272_zh-cn_topic_0000001389787297_row6223476444"><th class="cellrowborder" valign="top" width="17.22%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001820490272_zh-cn_topic_0000001389787297_p10223674448"><a name="zh-cn_topic_0000001820490272_zh-cn_topic_0000001389787297_p10223674448"></a><a name="zh-cn_topic_0000001820490272_zh-cn_topic_0000001389787297_p10223674448"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="15.340000000000002%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001820490272_zh-cn_topic_0000001389787297_p645511218169"><a name="zh-cn_topic_0000001820490272_zh-cn_topic_0000001389787297_p645511218169"></a><a name="zh-cn_topic_0000001820490272_zh-cn_topic_0000001389787297_p645511218169"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="67.44%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001820490272_zh-cn_topic_0000001389787297_p1922337124411"><a name="zh-cn_topic_0000001820490272_zh-cn_topic_0000001389787297_p1922337124411"></a><a name="zh-cn_topic_0000001820490272_zh-cn_topic_0000001389787297_p1922337124411"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row172853558203"><td class="cellrowborder" valign="top" width="17.22%" headers="mcps1.1.4.1.1 "><p id="p228575592010"><a name="p228575592010"></a><a name="p228575592010"></a>context</p>
</td>
<td class="cellrowborder" valign="top" width="15.340000000000002%" headers="mcps1.1.4.1.2 "><p id="p112851255142017"><a name="p112851255142017"></a><a name="p112851255142017"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.44%" headers="mcps1.1.4.1.3 "><p id="p928555522016"><a name="p928555522016"></a><a name="p928555522016"></a>TilingFunc注册上下文。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001820490272_zh-cn_topic_0000001389787297_row152234713443"><td class="cellrowborder" valign="top" width="17.22%" headers="mcps1.1.4.1.1 "><p id="p1693913355118"><a name="p1693913355118"></a><a name="p1693913355118"></a>...</p>
</td>
<td class="cellrowborder" valign="top" width="15.340000000000002%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001820490272_zh-cn_topic_0000001389787297_p15663137127"><a name="zh-cn_topic_0000001820490272_zh-cn_topic_0000001389787297_p15663137127"></a><a name="zh-cn_topic_0000001820490272_zh-cn_topic_0000001389787297_p15663137127"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.44%" headers="mcps1.1.4.1.3 "><p id="p010373710272"><a name="p010373710272"></a><a name="p010373710272"></a>可变长参数，模板参数的具体值，传入时需要与定义模板参数和模板参数组合的头文件中的模板参数顺序保持一致。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0000001867289945_zh-cn_topic_0000001389787297_section25791320141317"></a>

无

## 约束说明<a name="zh-cn_topic_0000001867289945_zh-cn_topic_0000001389787297_section19165124931511"></a>

无

## 调用示例<a name="zh-cn_topic_0000001867289945_zh-cn_topic_0000001389787297_section320753512363"></a>

```
#include "tiling_key_add_custom.h"
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    TilingDataTemplate tiling;
    uint32_t totalLength = context->GetInputShape(0)->GetOriginShape().GetShapeSize();
    ge::DataType dtype_x = context->GetInputDesc(0)->GetDataType();
    ge::DataType dtype_y = context->GetInputDesc(1)->GetDataType();
    ge::DataType dtype_z = context->GetOutputDesc(0)->GetDataType();
    uint32_t D_T_X = static_cast<int>(dtype_x), D_T_Y = static_cast<int>(dtype_y), D_T_Z = static_cast<int>(dtype_z), TILE_NUM = 1, IS_SPLIT = 0;
    if (totalLength < MIN_LENGTH_FOR_SPLIT) {
        IS_SPLIT = 0;
        TILE_NUM = 1;
    } else {
        IS_SPLIT = 1;
        TILE_NUM = DEFAULT_TILE_NUM;
    }
    context->SetBlockDim(NUM_BLOCKS);
    tiling.set_totalLength(totalLength);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    ASCENDC_TPL_SEL_PARAM(context, D_T_X, D_T_Y, D_T_Z, TILE_NUM, IS_SPLIT);
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
```

