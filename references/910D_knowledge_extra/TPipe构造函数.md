# TPipe构造函数<a name="ZH-CN_TOPIC_0000002554343471"></a>

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

构造用来管理内存和同步的TPipe对象。

## 函数原型<a name="section620mcpsimp"></a>

```
__aicore__ inline TPipe()
```

## 约束说明<a name="section633mcpsimp"></a>

-   避免TPipe在对象内创建和初始化，TPipe在对象内创建时，可能会影响编译器对对象内常量的优化，引起scalar性能劣化，具体原理请参考[避免TPipe在对象内创建和初始化](避免TPipe在对象内创建和初始化.md)。
-   TPipe对象同一时刻全局只能存在一份，同时定义多个TPipe对象，会出现卡死等随机行为。如果需要使用多个TPipe时，请先调用[Destroy](Destroy.md)接口释放前一个TPipe。

## 返回值说明<a name="section640mcpsimp"></a>

无

## 调用示例<a name="section642mcpsimp"></a>

```
template <typename ComputeT> class KernelExample {
public:
    __aicore__ inline KernelExample() {}
    __aicore__ inline void Init(..., TPipe* pipeIn)
    {
        ...
        pipe = pipeIn;
        pipe->InitBuffer(xxxBuf, BUFFER_NUM, xxxSize);
        ...
    }
private:
    ...
    TPipe* pipe;
    ...
};
extern "C" __global__ __aicore__ void example_kernel(...) {
    ...
    TPipe pipe;
    KernelExample<float> op;
    op.Init(..., &pipe);
    ...
}
```

