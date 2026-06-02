# GetUserWorkspace<a name="ZH-CN_TOPIC_0000002523344978"></a>

## 产品支持情况<a name="section1550532418810"></a>

<a name="zh-cn_topic_0000002554424517_table38301303189"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000002554424517_row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0000002554424517_p1883113061818"><a name="zh-cn_topic_0000002554424517_p1883113061818"></a><a name="zh-cn_topic_0000002554424517_p1883113061818"></a><span id="zh-cn_topic_0000002554424517_ph20833205312295"><a name="zh-cn_topic_0000002554424517_ph20833205312295"></a><a name="zh-cn_topic_0000002554424517_ph20833205312295"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="zh-cn_topic_0000002554424517_p783113012187"><a name="zh-cn_topic_0000002554424517_p783113012187"></a><a name="zh-cn_topic_0000002554424517_p783113012187"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000002554424517_row1272474920205"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000002554424517_p17301775812"><a name="zh-cn_topic_0000002554424517_p17301775812"></a><a name="zh-cn_topic_0000002554424517_p17301775812"></a><span id="zh-cn_topic_0000002554424517_ph2272194216543"><a name="zh-cn_topic_0000002554424517_ph2272194216543"></a><a name="zh-cn_topic_0000002554424517_ph2272194216543"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000002554424517_p37256491200"><a name="zh-cn_topic_0000002554424517_p37256491200"></a><a name="zh-cn_topic_0000002554424517_p37256491200"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section618mcpsimp"></a>

获取用户使用的workspace指针。workspace的具体介绍请参考[如何使用workspace](如何使用workspace.md)。Kernel直调开发方式下，如果未开启[HAVE\_WORKSPACE](基于样例工程完成Kernel直调.md#table481718169817)编译选项，框架不会自动设置系统workspace。如果使用了[Matmul Kernel侧接口](Matmul-Kernel侧接口.md)等需要系统workspace的高阶API，kernel侧需要通过[SetSysWorkSpace](SetSysWorkSpace.md)设置系统workspace，此时用户workspace需要通过该接口获取。

## 函数原型<a name="section620mcpsimp"></a>

```
__aicore__ inline GM_ADDR GetUserWorkspace(GM_ADDR workspace)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  接口参数说明

<a name="table1055216132132"></a>
<table><thead align="left"><tr id="row105531513121315"><th class="cellrowborder" valign="top" width="16.49%" id="mcps1.2.4.1.1"><p id="p5553171319138"><a name="p5553171319138"></a><a name="p5553171319138"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="11.93%" id="mcps1.2.4.1.2"><p id="p5553151313131"><a name="p5553151313131"></a><a name="p5553151313131"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="71.58%" id="mcps1.2.4.1.3"><p id="p655316136139"><a name="p655316136139"></a><a name="p655316136139"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row5553201314135"><td class="cellrowborder" valign="top" width="16.49%" headers="mcps1.2.4.1.1 "><p id="p8553813111314"><a name="p8553813111314"></a><a name="p8553813111314"></a>workspace</p>
</td>
<td class="cellrowborder" valign="top" width="11.93%" headers="mcps1.2.4.1.2 "><p id="p755318134134"><a name="p755318134134"></a><a name="p755318134134"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.58%" headers="mcps1.2.4.1.3 "><p id="p1185064715302"><a name="p1185064715302"></a><a name="p1185064715302"></a>传入workspace的指针，包括系统workspace和用户使用的workspace。</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section633mcpsimp"></a>

无

## 返回值说明<a name="section640mcpsimp"></a>

用户使用workspace指针。

