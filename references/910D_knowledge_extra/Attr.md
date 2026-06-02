# Attr<a name="ZH-CN_TOPIC_0000002554343887"></a>

## 功能说明<a name="zh-cn_topic_0000001705099477_zh-cn_topic_0000001576870901_zh-cn_topic_0000001575944081_section36583473819"></a>

注册算子属性参数。

当需要设置的参数不参与kernel侧计算时，可以将该参数注册为算子属性参数。

## 函数原型<a name="zh-cn_topic_0000001705099477_zh-cn_topic_0000001576870901_zh-cn_topic_0000001575944081_section13230182415108"></a>

```
OpAttrDef &Attr(const char *name)
```

## 参数说明<a name="zh-cn_topic_0000001705099477_zh-cn_topic_0000001576870901_zh-cn_topic_0000001575944081_section75395119104"></a>

<a name="zh-cn_topic_0000001705099477_zh-cn_topic_0000001576870901_zh-cn_topic_0000001575944081_table111938719446"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001705099477_zh-cn_topic_0000001576870901_zh-cn_topic_0000001575944081_row6223476444"><th class="cellrowborder" valign="top" width="17.22%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001705099477_zh-cn_topic_0000001576870901_zh-cn_topic_0000001575944081_p10223674448"><a name="zh-cn_topic_0000001705099477_zh-cn_topic_0000001576870901_zh-cn_topic_0000001575944081_p10223674448"></a><a name="zh-cn_topic_0000001705099477_zh-cn_topic_0000001576870901_zh-cn_topic_0000001575944081_p10223674448"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="15.340000000000002%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001705099477_zh-cn_topic_0000001576870901_zh-cn_topic_0000001575944081_p645511218169"><a name="zh-cn_topic_0000001705099477_zh-cn_topic_0000001576870901_zh-cn_topic_0000001575944081_p645511218169"></a><a name="zh-cn_topic_0000001705099477_zh-cn_topic_0000001576870901_zh-cn_topic_0000001575944081_p645511218169"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="67.44%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001705099477_zh-cn_topic_0000001576870901_zh-cn_topic_0000001575944081_p1922337124411"><a name="zh-cn_topic_0000001705099477_zh-cn_topic_0000001576870901_zh-cn_topic_0000001575944081_p1922337124411"></a><a name="zh-cn_topic_0000001705099477_zh-cn_topic_0000001576870901_zh-cn_topic_0000001575944081_p1922337124411"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001705099477_zh-cn_topic_0000001576870901_zh-cn_topic_0000001575944081_row152234713443"><td class="cellrowborder" valign="top" width="17.22%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001705099477_zh-cn_topic_0000001576870901_zh-cn_topic_0000001575944081_p318615392613"><a name="zh-cn_topic_0000001705099477_zh-cn_topic_0000001576870901_zh-cn_topic_0000001575944081_p318615392613"></a><a name="zh-cn_topic_0000001705099477_zh-cn_topic_0000001576870901_zh-cn_topic_0000001575944081_p318615392613"></a>name</p>
</td>
<td class="cellrowborder" valign="top" width="15.340000000000002%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001705099477_zh-cn_topic_0000001576870901_zh-cn_topic_0000001575944081_p320343694214"><a name="zh-cn_topic_0000001705099477_zh-cn_topic_0000001576870901_zh-cn_topic_0000001575944081_p320343694214"></a><a name="zh-cn_topic_0000001705099477_zh-cn_topic_0000001576870901_zh-cn_topic_0000001575944081_p320343694214"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.44%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001705099477_zh-cn_topic_0000001576870901_zh-cn_topic_0000001575944081_p453018873120"><a name="zh-cn_topic_0000001705099477_zh-cn_topic_0000001576870901_zh-cn_topic_0000001575944081_p453018873120"></a><a name="zh-cn_topic_0000001705099477_zh-cn_topic_0000001576870901_zh-cn_topic_0000001575944081_p453018873120"></a>算子属性名称。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0000001705099477_zh-cn_topic_0000001576870901_zh-cn_topic_0000001575944081_section25791320141317"></a>

算子属性定义，OpAttrDef请参考[OpAttrDef](OpAttrDef-167.md)。

## 约束说明<a name="zh-cn_topic_0000001705099477_zh-cn_topic_0000001576870901_zh-cn_topic_0000001575944081_section19165124931511"></a>

Attr属性名不能与以下python关键字及内置变量名相同，否则会导致未定义错误。

-   常见python关键字参考

    and、 as、 assert、 break、 class、 continue、 def、 del、 elif、 else、 except、 finally、 for、 from、 global、 if、 import、 in、 is、 lambda、 not、 or、 pass、 raise、 return、 try、 while、 with、 yield、 False、 None、 True、 nonlocal、 arg。

-   内置变量名

    \_\_inputs\_\_、 \_\_outputs\_\_、 \_\_attrs\_\_、 options、 bisheng、 bisheng\_path、 tikcpp\_path、 impl\_mode、 custom\_compile\_options、 custom\_all\_compile\_options、 soc\_version、 soc\_short、 custom\_compile\_options\_soc、 custom\_all\_compile\_options\_soc、 origin\_func\_name、 ascendc\_src\_dir\_ex、 ascendc\_src\_dir、 ascendc\_src\_file、 src、 op\_type、 code\_channel、 op\_info、 compile\_op、 get\_code\_channel、 result、 isinstance、 attr、 get\_current\_build\_config、 \_build\_args、 get\_dtype\_fmt\_options、 shutil、 os、 get\_kernel\_source、ascendc\_api\_version\_header\_path、ascendc\_api\_version\_file、ascendc\_api\_version、re。

