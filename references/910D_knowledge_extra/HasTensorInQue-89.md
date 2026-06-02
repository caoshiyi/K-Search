# HasTensorInQue<a name="ZH-CN_TOPIC_0000002523344502"></a>

## 产品支持情况<a name="section1550532418810"></a>

<a name="zh-cn_topic_0000002554423509_table38301303189"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000002554423509_row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0000002554423509_p1883113061818"><a name="zh-cn_topic_0000002554423509_p1883113061818"></a><a name="zh-cn_topic_0000002554423509_p1883113061818"></a><span id="zh-cn_topic_0000002554423509_ph20833205312295"><a name="zh-cn_topic_0000002554423509_ph20833205312295"></a><a name="zh-cn_topic_0000002554423509_ph20833205312295"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="zh-cn_topic_0000002554423509_p783113012187"><a name="zh-cn_topic_0000002554423509_p783113012187"></a><a name="zh-cn_topic_0000002554423509_p783113012187"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000002554423509_row1272474920205"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000002554423509_p17301775812"><a name="zh-cn_topic_0000002554423509_p17301775812"></a><a name="zh-cn_topic_0000002554423509_p17301775812"></a><span id="zh-cn_topic_0000002554423509_ph2272194216543"><a name="zh-cn_topic_0000002554423509_ph2272194216543"></a><a name="zh-cn_topic_0000002554423509_ph2272194216543"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000002554423509_p37256491200"><a name="zh-cn_topic_0000002554423509_p37256491200"></a><a name="zh-cn_topic_0000002554423509_p37256491200"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section618mcpsimp"></a>

查询Que中目前是否已有入队的Tensor。

## 函数原型<a name="section620mcpsimp"></a>

```
__aicore__ inline bool HasTensorInQue()
```

## 参数说明<a name="section622mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

该接口不支持[Tensor原地操作](如何使用Tensor原地操作提升算子性能.md)，即TQue的depth设置为0的场景。

## 返回值说明<a name="section640mcpsimp"></a>

-   true - 表示Queue中存在已入队的Tensor
-   false - 表示Queue为完全空闲

## 调用示例<a name="section642mcpsimp"></a>

```
// 根据VacantInQue判断当前que中是否有已入队的Tensor，当前que的深度为4，无内存Enque动作，返回为false
AscendC::TPipe pipe;
AscendC::TQueBind<AscendC::TPosition::VECOUT, AscendC::TPosition::GM, 4> que;
int num = 4;
int len = 1024;
pipe.InitBuffer(que, num, len);
bool ret = que.HasTensorInQue();
```

