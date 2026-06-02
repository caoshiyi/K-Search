# GetSsbufBaseAddr<a name="ZH-CN_TOPIC_0000002523344208"></a>

## 产品支持情况<a name="section11658125112438"></a>

<a name="table1165815116436"></a>
<table><thead align="left"><tr id="row165885144318"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p5658175124313"><a name="p5658175124313"></a><a name="p5658175124313"></a><span id="ph1865816515435"><a name="ph1865816515435"></a><a name="ph1865816515435"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p2065805124318"><a name="p2065805124318"></a><a name="p2065805124318"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row6658115154311"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p2658175111436"><a name="p2658175111436"></a><a name="p2658175111436"></a><span id="ph96581951124313"><a name="ph96581951124313"></a><a name="ph96581951124313"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p5658351144316"><a name="p5658351144316"></a><a name="p5658351144316"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section1966075164312"></a>

该接口用于获取SSBuffer的基地址。

## 函数原型<a name="section266175134310"></a>

```
__aicore__ inline __ssbuf__ void*  GetSsbufBaseAddr()
```

## 参数说明<a name="section766115164313"></a>

无

## 返回值说明<a name="section166616514437"></a>

返回指向SSBuffer基地址的指针。

## 约束说明<a name="section156613513435"></a>

1.SSBuffer中存在脏数据，读取时数据时不保证全为0。

2.AIC和AIV启动不同的任务时，不能访问SSBuffer。

3.访问超过最末端的地址存在异常。每个核在非MIX模式下运行时，可以独立占用1KB的空间（AIC，AIV0，AIV1各占据1KB）；或者在Mix模式下运行时共享整个3KB的空间\(AIC:AIV = 1:2\)。目前ssbuf的大小为3KB。

4.只支持通过读写指令32B,64B 的对齐访问。

## 调用示例<a name="section11661185118437"></a>

```
 __ssbuf__ void* ssbuf = GetSsbufBaseAddr();
```

