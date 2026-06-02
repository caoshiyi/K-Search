# AddAttr<a name="ZH-CN_TOPIC_0000002523344840"></a>

## 功能说明<a name="zh-cn_topic_0000001820490268_zh-cn_topic_0000001389787297_section36583473819"></a>

设置算子的属性以及对应值

## 函数原型<a name="zh-cn_topic_0000001820490268_zh-cn_topic_0000001389787297_section13230182415108"></a>

```
ContextBuilder &AddAttr(const std::string& attrName, int64_t attrValue)
ContextBuilder &AddAttr(const std::string& attrName, bool attrValue)
ContextBuilder &AddAttr(const std::string& attrName, const std::string& attrValue)
ContextBuilder &AddAttr(const std::string& attrName, float attrValue)
ContextBuilder &AddAttr(const std::string& attrName, const std::vector<float>& attrValue)
ContextBuilder &AddAttr(const std::string& attrName, const std::vector<bool>& attrValue)
ContextBuilder &AddAttr(const std::string& attrName, const std::vector<int64_t>& attrValue)
ContextBuilder &AddAttr(const std::string& attrName, const std::vector<std::string>& attrValue)
ContextBuilder &AddAttr(const std::string& attrName, const std::vector<std::vector<int64_t>>& attrValue)
```

## 参数说明<a name="zh-cn_topic_0000001820490268_zh-cn_topic_0000001389787297_section75395119104"></a>

<a name="zh-cn_topic_0000001820490268_zh-cn_topic_0000001389787297_table111938719446"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001820490268_zh-cn_topic_0000001389787297_row6223476444"><th class="cellrowborder" valign="top" width="17.22%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001820490268_zh-cn_topic_0000001389787297_p10223674448"><a name="zh-cn_topic_0000001820490268_zh-cn_topic_0000001389787297_p10223674448"></a><a name="zh-cn_topic_0000001820490268_zh-cn_topic_0000001389787297_p10223674448"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="15.340000000000002%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001820490268_zh-cn_topic_0000001389787297_p645511218169"><a name="zh-cn_topic_0000001820490268_zh-cn_topic_0000001389787297_p645511218169"></a><a name="zh-cn_topic_0000001820490268_zh-cn_topic_0000001389787297_p645511218169"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="67.44%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001820490268_zh-cn_topic_0000001389787297_p1922337124411"><a name="zh-cn_topic_0000001820490268_zh-cn_topic_0000001389787297_p1922337124411"></a><a name="zh-cn_topic_0000001820490268_zh-cn_topic_0000001389787297_p1922337124411"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001820490268_zh-cn_topic_0000001389787297_row152234713443"><td class="cellrowborder" valign="top" width="17.22%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001820490268_zh-cn_topic_0000001389787297_p8563195616313"><a name="zh-cn_topic_0000001820490268_zh-cn_topic_0000001389787297_p8563195616313"></a><a name="zh-cn_topic_0000001820490268_zh-cn_topic_0000001389787297_p8563195616313"></a>attrName</p>
</td>
<td class="cellrowborder" valign="top" width="15.340000000000002%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001820490268_zh-cn_topic_0000001389787297_p15663137127"><a name="zh-cn_topic_0000001820490268_zh-cn_topic_0000001389787297_p15663137127"></a><a name="zh-cn_topic_0000001820490268_zh-cn_topic_0000001389787297_p15663137127"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.44%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001820490268_zh-cn_topic_0000001389787297_p2684123934216"><a name="zh-cn_topic_0000001820490268_zh-cn_topic_0000001389787297_p2684123934216"></a><a name="zh-cn_topic_0000001820490268_zh-cn_topic_0000001389787297_p2684123934216"></a>算子的属性名称</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001820490268_row4927145645816"><td class="cellrowborder" valign="top" width="17.22%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001820490268_p5927145635816"><a name="zh-cn_topic_0000001820490268_p5927145635816"></a><a name="zh-cn_topic_0000001820490268_p5927145635816"></a>attrValue</p>
</td>
<td class="cellrowborder" valign="top" width="15.340000000000002%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001820490268_p1492795645810"><a name="zh-cn_topic_0000001820490268_p1492795645810"></a><a name="zh-cn_topic_0000001820490268_p1492795645810"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.44%" headers="mcps1.1.4.1.3 "><p id="p1490882610516"><a name="p1490882610516"></a><a name="p1490882610516"></a>算子属性值，支持int64_t、bool、std::string、float、std::vector&lt;float&gt;、std::vector&lt;int64_t&gt;、</p>
<p id="zh-cn_topic_0000001820490268_p1292745695816"><a name="zh-cn_topic_0000001820490268_p1292745695816"></a><a name="zh-cn_topic_0000001820490268_p1292745695816"></a>std::vector&lt;std::string&gt;、std::vector&lt;bool&gt;、std::vector&lt;std::vector&lt;int64_t&gt;&gt;类型的指定。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0000001820490268_zh-cn_topic_0000001389787297_section25791320141317"></a>

当前ContextBuilder的对象。

## 约束说明<a name="zh-cn_topic_0000001820490268_zh-cn_topic_0000001389787297_section19165124931511"></a>

AddAttr不支持重复添加同名的属性

## 调用示例<a name="zh-cn_topic_0000001820490268_zh-cn_topic_0000001389787297_section320753512363"></a>

```
context_ascendc::ContextBuilder builder;
auto builder
    .NodeIoNum(1,1)
    .IrInstanceNum({1})
    .AddAttr("attr_1", 1)
    .AddAttr("attr_2", true)
    .AddAttr("attr_3", "stringValue")
    .AddAttr("attr_4", 1.f)
    .AddAttr("attr_5", {1})
    .AddAttr("attr_6", {false})
    .AddAttr("attr_7", {"stringValue"})
    .AddAttr("attr_8", {1.f})
    .AddAttr("attr_9", {{1, 2}, {3, 4}})
```

