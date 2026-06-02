# GmFree<a name="ZH-CN_TOPIC_0000002523303904"></a>

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

## 功能说明<a name="zh-cn_topic_0000002000199397_zh-cn_topic_0000001592243993_section259105813316"></a>

进行核函数的CPU侧运行验证时，用于释放通过[GmAlloc](GmAlloc.md)申请的共享内存。

## 函数原型<a name="zh-cn_topic_0000002000199397_zh-cn_topic_0000001592243993_section2067518173415"></a>

```
void GmFree(void *ptr)
```

## 参数说明<a name="zh-cn_topic_0000002000199397_zh-cn_topic_0000001592243993_section158061867342"></a>

<a name="zh-cn_topic_0000002000199397_zh-cn_topic_0000001592243993_zh-cn_topic_0235751031_table33761356"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000002000199397_zh-cn_topic_0000001592243993_zh-cn_topic_0235751031_row27598891"><th class="cellrowborder" valign="top" width="16.49%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000002000199397_zh-cn_topic_0000001592243993_zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0000002000199397_zh-cn_topic_0000001592243993_zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0000002000199397_zh-cn_topic_0000001592243993_zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="11.93%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000002000199397_zh-cn_topic_0000001592243993_zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0000002000199397_zh-cn_topic_0000001592243993_zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0000002000199397_zh-cn_topic_0000001592243993_zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="71.58%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000002000199397_zh-cn_topic_0000001592243993_zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0000002000199397_zh-cn_topic_0000001592243993_zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0000002000199397_zh-cn_topic_0000001592243993_zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000002000199397_zh-cn_topic_0000001592243993_row42461942101815"><td class="cellrowborder" valign="top" width="16.49%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002000199397_zh-cn_topic_0000001592243993_p284425844311"><a name="zh-cn_topic_0000002000199397_zh-cn_topic_0000001592243993_p284425844311"></a><a name="zh-cn_topic_0000002000199397_zh-cn_topic_0000001592243993_p284425844311"></a>ptr</p>
</td>
<td class="cellrowborder" valign="top" width="11.93%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002000199397_zh-cn_topic_0000001592243993_p158449584436"><a name="zh-cn_topic_0000002000199397_zh-cn_topic_0000001592243993_p158449584436"></a><a name="zh-cn_topic_0000002000199397_zh-cn_topic_0000001592243993_p158449584436"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.58%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002000199397_zh-cn_topic_0000001592243993_p297233812230"><a name="zh-cn_topic_0000002000199397_zh-cn_topic_0000001592243993_p297233812230"></a><a name="zh-cn_topic_0000002000199397_zh-cn_topic_0000001592243993_p297233812230"></a>需要释放的共享内存的指针。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0000002000199397_zh-cn_topic_0000001592243993_section640mcpsimp"></a>

无

## 约束说明<a name="zh-cn_topic_0000002000199397_zh-cn_topic_0000001592243993_section794123819592"></a>

传入的指针必须是之前通过GmAlloc申请过的共享内存的指针。

## 调用示例<a name="zh-cn_topic_0000002000199397_zh-cn_topic_0000001592243993_section82241477610"></a>

```
AscendC::GmFree((void*)x);
```

