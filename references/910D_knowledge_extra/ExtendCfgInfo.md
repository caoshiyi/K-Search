# ExtendCfgInfo<a name="ZH-CN_TOPIC_0000002554423505"></a>

## 功能说明<a name="zh-cn_topic_0000001600884192_zh-cn_topic_0000001576875005_zh-cn_topic_0000001525424352_section36583473819"></a>

用于扩展算子相关参数配置，提供更灵活的参数配置能力。

## 函数原型<a name="zh-cn_topic_0000001600884192_zh-cn_topic_0000001576875005_zh-cn_topic_0000001525424352_section13230182415108"></a>

```
OpAICoreConfig &OpAICoreConfig::ExtendCfgInfo(const char *key, const char *value) 
```

## 参数说明<a name="zh-cn_topic_0000001600884192_zh-cn_topic_0000001576875005_zh-cn_topic_0000001525424352_section75395119104"></a>

**表 1**  参数说明

<a name="zh-cn_topic_0000001600884192_zh-cn_topic_0000001576875005_zh-cn_topic_0000001575944081_table111938719446"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001600884192_zh-cn_topic_0000001576875005_zh-cn_topic_0000001575944081_row6223476444"><th class="cellrowborder" valign="top" width="17.22%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001600884192_zh-cn_topic_0000001576875005_zh-cn_topic_0000001575944081_p10223674448"><a name="zh-cn_topic_0000001600884192_zh-cn_topic_0000001576875005_zh-cn_topic_0000001575944081_p10223674448"></a><a name="zh-cn_topic_0000001600884192_zh-cn_topic_0000001576875005_zh-cn_topic_0000001575944081_p10223674448"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="15.340000000000002%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001600884192_zh-cn_topic_0000001576875005_zh-cn_topic_0000001575944081_p645511218169"><a name="zh-cn_topic_0000001600884192_zh-cn_topic_0000001576875005_zh-cn_topic_0000001575944081_p645511218169"></a><a name="zh-cn_topic_0000001600884192_zh-cn_topic_0000001576875005_zh-cn_topic_0000001575944081_p645511218169"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="67.44%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001600884192_zh-cn_topic_0000001576875005_zh-cn_topic_0000001575944081_p1922337124411"><a name="zh-cn_topic_0000001600884192_zh-cn_topic_0000001576875005_zh-cn_topic_0000001575944081_p1922337124411"></a><a name="zh-cn_topic_0000001600884192_zh-cn_topic_0000001576875005_zh-cn_topic_0000001575944081_p1922337124411"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001600884192_zh-cn_topic_0000001576875005_zh-cn_topic_0000001575944081_row152234713443"><td class="cellrowborder" valign="top" width="17.22%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001600884192_zh-cn_topic_0000001576875005_zh-cn_topic_0000001575944081_p318615392613"><a name="zh-cn_topic_0000001600884192_zh-cn_topic_0000001576875005_zh-cn_topic_0000001575944081_p318615392613"></a><a name="zh-cn_topic_0000001600884192_zh-cn_topic_0000001576875005_zh-cn_topic_0000001575944081_p318615392613"></a>key</p>
</td>
<td class="cellrowborder" valign="top" width="15.340000000000002%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001600884192_zh-cn_topic_0000001576875005_zh-cn_topic_0000001575944081_p320343694214"><a name="zh-cn_topic_0000001600884192_zh-cn_topic_0000001576875005_zh-cn_topic_0000001575944081_p320343694214"></a><a name="zh-cn_topic_0000001600884192_zh-cn_topic_0000001576875005_zh-cn_topic_0000001575944081_p320343694214"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.44%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001600884192_p1979514644510"><a name="zh-cn_topic_0000001600884192_p1979514644510"></a><a name="zh-cn_topic_0000001600884192_p1979514644510"></a>配置项，如“aclnnSupport.value”。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001600884192_row1271485544013"><td class="cellrowborder" valign="top" width="17.22%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001600884192_p1271465544017"><a name="zh-cn_topic_0000001600884192_p1271465544017"></a><a name="zh-cn_topic_0000001600884192_p1271465544017"></a>value</p>
</td>
<td class="cellrowborder" valign="top" width="15.340000000000002%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001600884192_p1371410554404"><a name="zh-cn_topic_0000001600884192_p1371410554404"></a><a name="zh-cn_topic_0000001600884192_p1371410554404"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.44%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001600884192_p16436143194216"><a name="zh-cn_topic_0000001600884192_p16436143194216"></a><a name="zh-cn_topic_0000001600884192_p16436143194216"></a>配置项key对应的取值，如“aclnnSupport.value”，可以填充“support_aclnn”或者“aclnn_only”。</p>
</td>
</tr>
</tbody>
</table>

ExtendCfgInfo支持的参数如下表：

**表 2**  ExtendCfgInfo支持配置的参数

<a name="table57844329252"></a>
<table><thead align="left"><tr id="row07842329252"><th class="cellrowborder" valign="top" width="16.919999999999998%" id="mcps1.2.3.1.1"><p id="p1778416325253"><a name="p1778416325253"></a><a name="p1778416325253"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="83.08%" id="mcps1.2.3.1.2"><p id="p107841532162519"><a name="p107841532162519"></a><a name="p107841532162519"></a>功能介绍</p>
</th>
</tr>
</thead>
<tbody><tr id="row978453211250"><td class="cellrowborder" valign="top" width="16.919999999999998%" headers="mcps1.2.3.1.1 "><p id="p14784123216256"><a name="p14784123216256"></a><a name="p14784123216256"></a>aclnnSupport.value</p>
</td>
<td class="cellrowborder" valign="top" width="83.08%" headers="mcps1.2.3.1.2 "><a name="ul18784732192518"></a><a name="ul18784732192518"></a><ul id="ul18784732192518"><li>support_aclnn：此模式下，静态Shape场景中该算子通过模型下沉执行，动态Shape场景则在Host侧调用fallback函数下发算子。如果调用了<a href="EnableFallBack.md">EnableFallBack</a>则默认采用该模式。<pre class="screen" id="screen1478483215256"><a name="screen1478483215256"></a><a name="screen1478483215256"></a>// 如下为动态Shape场景的示例
OpAICoreConfig aicore_config;
aicore_config.DynamicShapeSupportFlag(true)   // 动态Shape场景需要设置DynamicShapeSupportFlag为true
             .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");
this-&gt;AICore().AddConfig("ascendxxx", aicore_config);

// 如下为静态Shape场景的示例
OpAICoreConfig aicore_config;
aicore_config.DynamicCompileStaticFlag(true)  // 静态Shape场景需要设置DynamicCompileStaticFlag为true
             .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");
this-&gt;AICore().AddConfig("ascendxxx", aicore_config);</pre>
</li><li>aclnn_only：此模式下，动静态Shape场景中该算子均以fallback形式下发。不建议用户使用该模式，后续版本待废弃。<pre class="screen" id="screen137851032142517"><a name="screen137851032142517"></a><a name="screen137851032142517"></a>// 如下为动态Shape场景的示例
OpAICoreConfig aicore_config;
aicore_config.DynamicShapeSupportFlag(true) // 动态Shape场景需要设置DynamicShapeSupportFlag为true
			 .ExtendCfgInfo("aclnnSupport.value", "aclnn_only");
this-&gt;AICore().AddConfig("ascendxxx", aicore_config);
// 如下为静态Shape场景的示例
OpAICoreConfig aicore_config;              // 无需配置DynamicCompileStaticFlag/DynamicShapeSupportFlag，算子均以fallback形式下发，即动态Shape模型的下发方式
aicore_config.ExtendCfgInfo("aclnnSupport.value", "aclnn_only");
this-&gt;AICore().AddConfig("ascendxxx", aicore_config);</pre>
</li></ul>
<p id="p760632713326"><a name="p760632713326"></a><a name="p760632713326"></a>关于fallback下发算子的详细介绍请参考<span id="ph1957516252014"><a name="ph1957516252014"></a><a name="ph1957516252014"></a>《图模式开发指南》</span>中的<span id="ph20215105411516"><a name="ph20215105411516"></a><a name="ph20215105411516"></a>“自定义算子入图开发 &gt; 基于fallback形式下发算子”</span>章节。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0000001600884192_zh-cn_topic_0000001576875005_zh-cn_topic_0000001525424352_section25791320141317"></a>

OpAICoreConfig算子定义，请参考[OpAICoreConfig](OpAICoreConfig.md)。

## 约束说明<a name="zh-cn_topic_0000001600884192_zh-cn_topic_0000001576875005_zh-cn_topic_0000001525424352_section19165124931511"></a>

无

