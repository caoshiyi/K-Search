# FreeAllEvent<a name="ZH-CN_TOPIC_0000002554423697"></a>

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

释放队列中申请的所有同步事件。队列分配的Buffer关联着同步事件的eventID，因为同步事件的数量有限制，如果同时使用的队列Buffer数量超过限制，将无法继续申请队列，使用本接口释放队列中的事件后，可以再次申请队列。详细介绍请参考[TQue Buffer限制](TQue简介.md#section466543213575)。

## 函数原型<a name="section620mcpsimp"></a>

```
__aicore__ inline void FreeAllEvent()
```

## 参数说明<a name="section622mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

该接口不支持[Tensor原地操作](如何使用Tensor原地操作提升算子性能.md)，即TQue的depth设置为0的场景。

## 返回值说明<a name="section640mcpsimp"></a>

无

## 调用示例<a name="section642mcpsimp"></a>

```
// 接口: DeQue Tensor
AscendC::TPipe pipe;
AscendC::TQueBind<AscendC::TPosition::VECOUT, AscendC::TPosition::GM, 4> que;
int num = 4;
int len = 1024;
pipe.InitBuffer(que, num, len);
AscendC::LocalTensor<half> tensor1 = que.AllocTensor<half>();
que.EnQue(tensor1);
tensor1 = que.DeQue<half>(); // 将tensor从VECOUT的Queue中搬出
que.FreeTensor<half>(tensor1);
que.FreeAllEvent();
```

