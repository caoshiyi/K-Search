# Cast（多类型转float）<a name="ZH-CN_TOPIC_0000002523304600"></a>

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

将输入数据转换为float类型。

## 函数原型<a name="section620mcpsimp"></a>

-   bfloat16\_t类型转换为float类型

    ```
    __aicore__ inline float Cast(const bfloat16_t& bVal)
    ```

-   支持多种数据类型转换为float类型

    ```
    
    template <typename T, typename U = float,
              typename = Std::enable_if_t<
              (Std::is_same<T, bfloat16_t>::value || Std::is_same<T, hifloat8_t>::value ||
               Std::is_same<T, fp8_e5m2_t>::value || Std::is_same<T, fp8_e4m3fn_t>::value ||
               Std::is_same<T, fp4x2_e1m2_t>::value || Std::is_same<T, fp4x2_e2m1_t>::value), 
              void>>
    __aicore__ constexpr inline U Cast(T bVal)
    ```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table4835205712588"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001429830437_row118356578583"><th class="cellrowborder" valign="top" width="16.28%" id="mcps1.2.3.1.1"><p id="zh-cn_topic_0000001429830437_p48354572582"><a name="zh-cn_topic_0000001429830437_p48354572582"></a><a name="zh-cn_topic_0000001429830437_p48354572582"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="83.72%" id="mcps1.2.3.1.2"><p id="zh-cn_topic_0000001429830437_p583535795817"><a name="zh-cn_topic_0000001429830437_p583535795817"></a><a name="zh-cn_topic_0000001429830437_p583535795817"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001429830437_row1835857145817"><td class="cellrowborder" valign="top" width="16.28%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001429830437_p5835457165816"><a name="zh-cn_topic_0000001429830437_p5835457165816"></a><a name="zh-cn_topic_0000001429830437_p5835457165816"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="83.72%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001429830437_p168351657155818"><a name="zh-cn_topic_0000001429830437_p168351657155818"></a><a name="zh-cn_topic_0000001429830437_p168351657155818"></a>操作数数据类型。</p>
<p id="p33891341206"><a name="p33891341206"></a><a name="p33891341206"></a><span id="ph18507114311268"><a name="ph18507114311268"></a><a name="ph18507114311268"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：fp4x2_e2m1_t、fp4x2_e1m2_t、hifloat8_t、fp8_e5m2_t、fp8_e4m3fn_t、bfloat16_t。</p>
<p id="p378823974412"><a name="p378823974412"></a><a name="p378823974412"></a>fp4x2_e1m2_t和fp4x2_e2m1_t类型为了满足1byte数据大小，构造时由两个四位标量数据拼接生成。转换时，被转换的是fp4x2_e1m2_t和fp4x2_e2m1_t标量数据中低位4bit的值，即[0:3]位。</p>
</td>
</tr>
<tr id="row178173845414"><td class="cellrowborder" valign="top" width="16.28%" headers="mcps1.2.3.1.1 "><p id="p1578119383542"><a name="p1578119383542"></a><a name="p1578119383542"></a><span id="ph144645408546"><a name="ph144645408546"></a><a name="ph144645408546"></a>U</span></p>
</td>
<td class="cellrowborder" valign="top" width="83.72%" headers="mcps1.2.3.1.2 "><p id="p3781193865418"><a name="p3781193865418"></a><a name="p3781193865418"></a><span id="ph1098295175410"><a name="ph1098295175410"></a><a name="ph1098295175410"></a>返回值数据类型，</span><span id="ph122418431568"><a name="ph122418431568"></a><a name="ph122418431568"></a>仅支持float</span><span id="ph3163164845613"><a name="ph3163164845613"></a><a name="ph3163164845613"></a>。</span></p>
</td>
</tr>
</tbody>
</table>

**表 2**  接口参数说明

<a name="table18368155193919"></a>
<table><thead align="left"><tr id="row1036805543911"><th class="cellrowborder" valign="top" width="16.38163816381638%" id="mcps1.2.4.1.1"><p id="p1836835511393"><a name="p1836835511393"></a><a name="p1836835511393"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="10.861086108610861%" id="mcps1.2.4.1.2"><p id="p10368255163915"><a name="p10368255163915"></a><a name="p10368255163915"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="72.75727572757276%" id="mcps1.2.4.1.3"><p id="p436875573911"><a name="p436875573911"></a><a name="p436875573911"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row1436825518395"><td class="cellrowborder" valign="top" width="16.38163816381638%" headers="mcps1.2.4.1.1 "><p id="p941862411595"><a name="p941862411595"></a><a name="p941862411595"></a>bVal</p>
</td>
<td class="cellrowborder" valign="top" width="10.861086108610861%" headers="mcps1.2.4.1.2 "><p id="p941792465918"><a name="p941792465918"></a><a name="p941792465918"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.75727572757276%" headers="mcps1.2.4.1.3 "><p id="p14161124195918"><a name="p14161124195918"></a><a name="p14161124195918"></a>待转换的标量数据。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

转换后的float类型标量数据。

## 约束说明<a name="section633mcpsimp"></a>

无

## 调用示例<a name="section19372434133520"></a>

完整的算子样例参考：[cast样例](https://gitcode.com/cann/asc-devkit/tree/master/examples/02_features/16_scalar_computation/cast)。

```

// 将srcLocal的第一个元素通过Cast转换成float类型，并将结果赋值给fValue
float fValue = AscendC::Cast(srcLocal.GetValue(0));
// 将标量fValue 填充到dstLocal的前 32 个位置
AscendC::Duplicate(dstLocal, fValue, 32);
```

结果示例如下：

```
输入数据(srcLocal):
[21.3750, 74.0000, 57.2500,  ...,  2.4062, 72.5000, 67.5000]
输出数据(dstLocal):
[21.375 21.375 21.375 ... 21.375 21.375 21.375]
```

```
// 如下是一个输入类型fp4x2_e1m2_t的示例：
float fValue = AscendC::Cast<T, float>(srcLocal.GetValue(0));
AscendC::Duplicate(dstLocal, fValue, bufferSize);

// 输入数据(srcLocal)（二进制表达）: [0b10001111 0b10001111 0b10001111 ...]
// 输出数据(dstLocal): [-1.75 -1.75 -1.75 ...]
```

