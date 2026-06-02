# PipeBarrier\(ISASI\)<a name="ZH-CN_TOPIC_0000002554423567"></a>

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

阻塞相同流水，具有数据依赖的相同流水之间需要插入此同步。

## 函数原型<a name="section620mcpsimp"></a>

```
template <pipe_t pipe>
__aicore__ inline void PipeBarrier()
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="zh-cn_topic_0235751031_table33761356"></a>
<table><thead align="left"><tr id="zh-cn_topic_0235751031_row27598891"><th class="cellrowborder" valign="top" width="20.61%" id="mcps1.2.3.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="79.39%" id="mcps1.2.3.1.2"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row42461942101815"><td class="cellrowborder" valign="top" width="20.61%" headers="mcps1.2.3.1.1 "><p id="p479605232211"><a name="p479605232211"></a><a name="p479605232211"></a>pipe</p>
</td>
<td class="cellrowborder" valign="top" width="79.39%" headers="mcps1.2.3.1.2 "><p id="p1179555214221"><a name="p1179555214221"></a><a name="p1179555214221"></a>模板参数，表示阻塞的流水类别。</p>
<p id="p315334751312"><a name="p315334751312"></a><a name="p315334751312"></a>支持的流水参考<a href="同步控制简介.md#section1272612276459">硬件流水类型</a>。</p>
<p id="p1982544516362"><a name="p1982544516362"></a><a name="p1982544516362"></a>如果不关注流水类别，希望阻塞所有流水，可以传入PIPE_ALL。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

Scalar流水之间的同步由硬件自动保证，调用PipeBarrier<PIPE\_S\>\(\)会引发硬件错误。

## 调用示例<a name="section837496171220"></a>

如下示例，Mul指令的输入dst0Local是Add指令的输出，两个矢量运算指令产生依赖，需要插入PipeBarrier保证两条指令的执行顺序。

注：仅作为示例参考，开启自动同步（Kernel直调算子工程和自定义算子开发工程已默认开启）的情况下，编译器自动插入PIPE\_V同步，无需开发者手动插入。

**图 1**  Mul指令和Add指令是串行关系，必须等待Add指令执行完成后，才能执行Mul指令。<a name="fig1359216580459"></a>  
<!-- img2text -->
```text
AscendC::LocalTensor<half> src0Local;
AscendC::LocalTensor<half> src1Local;
AscendC::LocalTensor<half> src2Local;

┌──────────────────────────────────────────────────────────────┐
│ Add(dst0Local, src0Local, src1Local, 512);                  │
└──────────────────────────────────────────────────────────────┘
                              │
                              │
                            PIPE_V
                              │
                              ↓
┌──────────────────────────────────────────────────────────────┐
│ Mul(dst1Local, dst0Local, src2Local, 512);                  │
└──────────────────────────────────────────────────────────────┘
```

```
AscendC::LocalTensor<half> src0Local;
AscendC::LocalTensor<half> src1Local;
AscendC::LocalTensor<half> src2Local;
AscendC::LocalTensor<half> dst0Local;
AscendC::LocalTensor<half> dst1Local;

AscendC::Add(dst0Local, src0Local, src1Local, 512);
AscendC::PipeBarrier<PIPE_V>();
AscendC::Mul(dst1Local, dst0Local, src2Local, 512);
```

