# TilingData<a name="ZH-CN_TOPIC_0000002554343663"></a>

## 功能说明<a name="zh-cn_topic_0000001820650068_zh-cn_topic_0000001389787297_section36583473819"></a>

传入指向TilingData的void\* 指针

## 函数原型<a name="zh-cn_topic_0000001820650068_zh-cn_topic_0000001389787297_section13230182415108"></a>

```
ContextBuilder &TilingData(void *tilingData)
```

## 参数说明<a name="zh-cn_topic_0000001820650068_zh-cn_topic_0000001389787297_section75395119104"></a>

<a name="zh-cn_topic_0000001820650068_zh-cn_topic_0000001389787297_table111938719446"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001820650068_zh-cn_topic_0000001389787297_row6223476444"><th class="cellrowborder" valign="top" width="17.22%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001820650068_zh-cn_topic_0000001389787297_p10223674448"><a name="zh-cn_topic_0000001820650068_zh-cn_topic_0000001389787297_p10223674448"></a><a name="zh-cn_topic_0000001820650068_zh-cn_topic_0000001389787297_p10223674448"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="15.340000000000002%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001820650068_zh-cn_topic_0000001389787297_p645511218169"><a name="zh-cn_topic_0000001820650068_zh-cn_topic_0000001389787297_p645511218169"></a><a name="zh-cn_topic_0000001820650068_zh-cn_topic_0000001389787297_p645511218169"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="67.44%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001820650068_zh-cn_topic_0000001389787297_p1922337124411"><a name="zh-cn_topic_0000001820650068_zh-cn_topic_0000001389787297_p1922337124411"></a><a name="zh-cn_topic_0000001820650068_zh-cn_topic_0000001389787297_p1922337124411"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001820650068_zh-cn_topic_0000001389787297_row152234713443"><td class="cellrowborder" valign="top" width="17.22%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001820650068_zh-cn_topic_0000001389787297_p8563195616313"><a name="zh-cn_topic_0000001820650068_zh-cn_topic_0000001389787297_p8563195616313"></a><a name="zh-cn_topic_0000001820650068_zh-cn_topic_0000001389787297_p8563195616313"></a>tilingData</p>
</td>
<td class="cellrowborder" valign="top" width="15.340000000000002%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001820650068_zh-cn_topic_0000001389787297_p15663137127"><a name="zh-cn_topic_0000001820650068_zh-cn_topic_0000001389787297_p15663137127"></a><a name="zh-cn_topic_0000001820650068_zh-cn_topic_0000001389787297_p15663137127"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.44%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001820650068_zh-cn_topic_0000001389787297_p2684123934216"><a name="zh-cn_topic_0000001820650068_zh-cn_topic_0000001389787297_p2684123934216"></a><a name="zh-cn_topic_0000001820650068_zh-cn_topic_0000001389787297_p2684123934216"></a>指向TilingData类的void*指针</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0000001820650068_zh-cn_topic_0000001389787297_section25791320141317"></a>

当前ContextBuilder的对象。

## 约束说明<a name="zh-cn_topic_0000001820650068_zh-cn_topic_0000001389787297_section19165124931511"></a>

由于TilingContext与KernelContext、TilingParseContext内部数据排序不同，TilingData\(\)只支持以调用BuildTilingContext\(\)为前提来使用；其他场景建议用Outputs接口，否则发生未定义行为。

## 调用示例<a name="zh-cn_topic_0000001820650068_zh-cn_topic_0000001389787297_section320753512363"></a>

```
void AddTilingData(void *tilingData)
{
    ......
    auto kernelContextHolder = context_ascendc::ContextBuilder()
                                    .TilingData(tilingData)
                                    .BuildTilingContext();
    ......
}
```

