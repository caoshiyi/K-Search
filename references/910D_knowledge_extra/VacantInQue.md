# VacantInQue<a name="ZH-CN_TOPIC_0000002523344084"></a>

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

## 功能说明<a name="section618mcpsimp"></a>

查询队列是否已满。

## 函数原型<a name="section620mcpsimp"></a>

```
__aicore__ inline bool VacantInQue()
```

## 参数说明<a name="section622mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

该接口不支持[Tensor原地操作](如何使用Tensor原地操作提升算子性能.md)，即TQue的depth设置为0的场景。

## 返回值说明<a name="section640mcpsimp"></a>

-   true - 表示Queue未满，可以继续Enque操作
-   false - 表示Queue已满，不可以继续入队

## 调用示例<a name="section642mcpsimp"></a>

```
// 根据VacantInQue判断当前que是否已满，设置当前队列深度为4
AscendC::TPipe pipe;
AscendC::TQue<AscendC::TPosition::VECOUT, 4> que;
int num = 10;
int len = 1024;
pipe.InitBuffer(que, num, len);
bool ret = que.VacantInQue(); // 返回为true 
AscendC::LocalTensor<half> tensor1 = que.AllocTensor<half>();
AscendC::LocalTensor<half> tensor2 = que.AllocTensor<half>();
AscendC::LocalTensor<half> tensor3 = que.AllocTensor<half>();
AscendC::LocalTensor<half> tensor4 = que.AllocTensor<half>();
AscendC::LocalTensor<half> tensor5 = que.AllocTensor<half>();
que.EnQue(tensor1);// 将tensor1加入VECOUT的Queue中
que.EnQue(tensor2);// 将tensor2加入VECOUT的Queue中
que.EnQue(tensor3);// 将tensor3加入VECOUT的Queue中
que.EnQue(tensor4);// 将tensor4加入VECOUT的Queue中
ret = que.VacantInQue(); // 返回为false, 继续入队操作（Enque）将报错
```

