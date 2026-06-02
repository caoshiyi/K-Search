# Unlock<a name="ZH-CN_TOPIC_0000002523303800"></a>

## 产品支持情况<a name="section73648168211"></a>

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

## 功能说明<a name="zh-cn_topic_0000002078486173_zh-cn_topic_0000001576727153_zh-cn_topic_0000001389787297_section36583473819"></a>

直到前置当前流水指令退出后，根据MutexID释放对应Mutex。

## 函数原型<a name="zh-cn_topic_0000002078486173_zh-cn_topic_0000001576727153_zh-cn_topic_0000001389787297_section13230182415108"></a>

```
template <pipe_t pipe>
static __aicore__ inline void Unlock(MutexID id)
```

## 参数说明<a name="zh-cn_topic_0000002078486173_zh-cn_topic_0000001576727153_zh-cn_topic_0000001389787297_section75395119104"></a>

**表 1**  模板参数说明

<a name="table1965616488585"></a>
<table><thead align="left"><tr id="row865644816588"><th class="cellrowborder" valign="top" width="18.29%" id="mcps1.2.3.1.1"><p id="p765612483583"><a name="p765612483583"></a><a name="p765612483583"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="81.71000000000001%" id="mcps1.2.3.1.2"><p id="p1657204895812"><a name="p1657204895812"></a><a name="p1657204895812"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row865704865817"><td class="cellrowborder" valign="top" width="18.29%" headers="mcps1.2.3.1.1 "><p id="p8657048165815"><a name="p8657048165815"></a><a name="p8657048165815"></a>pipe</p>
</td>
<td class="cellrowborder" valign="top" width="81.71000000000001%" headers="mcps1.2.3.1.2 "><p id="p1465784819589"><a name="p1465784819589"></a><a name="p1465784819589"></a>模板参数，表示流水类别。</p>
<p id="p66571048135814"><a name="p66571048135814"></a><a name="p66571048135814"></a>支持的流水参考<a href="同步控制简介.md#section1272612276459">硬件流水类型</a>。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  参数说明

<a name="table103102222579"></a>
<table><thead align="left"><tr id="row631032216571"><th class="cellrowborder" valign="top" width="18.54%" id="mcps1.2.4.1.1"><p id="p231082295719"><a name="p231082295719"></a><a name="p231082295719"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="10.05%" id="mcps1.2.4.1.2"><p id="p2031014221576"><a name="p2031014221576"></a><a name="p2031014221576"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="71.41%" id="mcps1.2.4.1.3"><p id="p103112022105713"><a name="p103112022105713"></a><a name="p103112022105713"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row83111722195716"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p131119221575"><a name="p131119221575"></a><a name="p131119221575"></a>id</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p63110229572"><a name="p63110229572"></a><a name="p63110229572"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p9311822145713"><a name="p9311822145713"></a><a name="p9311822145713"></a>进行流水同步管理的MutexID。</p>
<p id="p195452267225"><a name="p195452267225"></a><a name="p195452267225"></a>该id可通过用户自定义（范围为0-27）或者通过<a href="AllocMutexID-(ISASI).md">AllocMutexID/ReleaseMutexID</a>进行申请释放。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0000002078486173_zh-cn_topic_0000001576727153_zh-cn_topic_0000001389787297_section25791320141317"></a>

无

## 约束说明<a name="zh-cn_topic_0000002078486173_zh-cn_topic_0000001576727153_zh-cn_topic_0000001389787297_section19165124931511"></a>

-   用户在使用Lock/Unlock时且自定义MutexID情况时，禁止同时使用[TQue](TQue.md)、[TQueBind](TQueBind.md)、[TBufPool](TBufPool.md)中的相关接口。
-   对于同一个MutexID，必须按照Lock/Unlock配套使用，且指定的pipe也需相同，即当且只有完成一个流水的Lock/Unlock之后，才能进行其余流水的操作。
-   相同流水之间存在数据依赖的场景，这种情况建议使用[PipeBarrier](PipeBarrier(ISASI).md)接口。

## 调用示例<a name="zh-cn_topic_0000002078486173_zh-cn_topic_0000001576727153_zh-cn_topic_0000001389787297_section320753512363"></a>

参考[调用示例](Lock.md#zh-cn_topic_0000002078486173_zh-cn_topic_0000001576727153_zh-cn_topic_0000001389787297_section320753512363)。

