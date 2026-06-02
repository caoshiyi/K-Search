# KernelRunContextHolder结构定义<a name="ZH-CN_TOPIC_0000002554423715"></a>

## 功能说明<a name="zh-cn_topic_0000001872423361_zh-cn_topic_0000001441184464_section618mcpsimp"></a>

该结构体为ContextBuilder类最终的构造结果，可通过指定的接口获取内部算子信息或获取KernelContext类的对象。

## 函数原型<a name="zh-cn_topic_0000001872423361_zh-cn_topic_0000001441184464_section620mcpsimp"></a>

```
struct KernelRunContextHolder {
    KernelRunContextHolder();
    ~KernelRunContextHolder();
    template<typename T>
    T *GetContext() const
    {
        return reinterpret_cast<T*>(context);
    }
    gert::ComputeNodeInfo *MutableComputeNodeInfo()
    {
        return reinterpret_cast<gert::ComputeNodeInfo *>(computeNodeExtendHolder.get());
    }
    std::unique_ptr<ValueHolderImpl> valueHolder;
    std::unique_ptr<uint8_t[]> computeNodeExtendHolder;
    KernelRunContext *context {nullptr};
};
```

## 函数说明<a name="zh-cn_topic_0000001872423361_zh-cn_topic_0000001441184464_section622mcpsimp"></a>

**表 1**  函数说明

<a name="zh-cn_topic_0000001872423361_zh-cn_topic_0000001441184464_table18149577913"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001872423361_zh-cn_topic_0000001441184464_row61411571196"><th class="cellrowborder" valign="top" width="15.409999999999998%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001872423361_zh-cn_topic_0000001441184464_p2093713281104"><a name="zh-cn_topic_0000001872423361_zh-cn_topic_0000001441184464_p2093713281104"></a><a name="zh-cn_topic_0000001872423361_zh-cn_topic_0000001441184464_p2093713281104"></a>函数名称</p>
</th>
<th class="cellrowborder" valign="top" width="21.34%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001872423361_zh-cn_topic_0000001441184464_p1593811282101"><a name="zh-cn_topic_0000001872423361_zh-cn_topic_0000001441184464_p1593811282101"></a><a name="zh-cn_topic_0000001872423361_zh-cn_topic_0000001441184464_p1593811282101"></a>入参说明</p>
</th>
<th class="cellrowborder" valign="top" width="63.24999999999999%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001872423361_zh-cn_topic_0000001441184464_p393813285106"><a name="zh-cn_topic_0000001872423361_zh-cn_topic_0000001441184464_p393813285106"></a><a name="zh-cn_topic_0000001872423361_zh-cn_topic_0000001441184464_p393813285106"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001872423361_zh-cn_topic_0000001441184464_row8906103284616"><td class="cellrowborder" valign="top" width="15.409999999999998%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001872423361_zh-cn_topic_0000001441184464_p39061832114616"><a name="zh-cn_topic_0000001872423361_zh-cn_topic_0000001441184464_p39061832114616"></a><a name="zh-cn_topic_0000001872423361_zh-cn_topic_0000001441184464_p39061832114616"></a>GetContext</p>
</td>
<td class="cellrowborder" valign="top" width="21.34%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001872423361_p945293784316"><a name="zh-cn_topic_0000001872423361_p945293784316"></a><a name="zh-cn_topic_0000001872423361_p945293784316"></a>无</p>
</td>
<td class="cellrowborder" valign="top" width="63.24999999999999%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001872423361_p7697185182511"><a name="zh-cn_topic_0000001872423361_p7697185182511"></a><a name="zh-cn_topic_0000001872423361_p7697185182511"></a>获取context成员变量转化为模板T的指针，T可选值为KernelContext以及它的子类如TilingContext</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001872423361_zh-cn_topic_0000001441184464_row433315311399"><td class="cellrowborder" valign="top" width="15.409999999999998%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001872423361_p69171825453"><a name="zh-cn_topic_0000001872423361_p69171825453"></a><a name="zh-cn_topic_0000001872423361_p69171825453"></a>MutableComputeNodeInfo</p>
</td>
<td class="cellrowborder" valign="top" width="21.34%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001872423361_p1591714255514"><a name="zh-cn_topic_0000001872423361_p1591714255514"></a><a name="zh-cn_topic_0000001872423361_p1591714255514"></a>无</p>
</td>
<td class="cellrowborder" valign="top" width="63.24999999999999%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001872423361_p149168253519"><a name="zh-cn_topic_0000001872423361_p149168253519"></a><a name="zh-cn_topic_0000001872423361_p149168253519"></a>返回构造的gert::ComputeNodeInfo类指针</p>
</td>
</tr>
</tbody>
</table>

**表 2**  变量说明

<a name="zh-cn_topic_0000001872423361_table161170462134"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001872423361_row511724611310"><th class="cellrowborder" valign="top" width="50%" id="mcps1.2.3.1.1"><p id="zh-cn_topic_0000001872423361_p91171467133"><a name="zh-cn_topic_0000001872423361_p91171467133"></a><a name="zh-cn_topic_0000001872423361_p91171467133"></a>变量名称</p>
</th>
<th class="cellrowborder" valign="top" width="50%" id="mcps1.2.3.1.2"><p id="zh-cn_topic_0000001872423361_p61171646141315"><a name="zh-cn_topic_0000001872423361_p61171646141315"></a><a name="zh-cn_topic_0000001872423361_p61171646141315"></a>变量含义</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001872423361_row12117104671311"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001872423361_p13118746101310"><a name="zh-cn_topic_0000001872423361_p13118746101310"></a><a name="zh-cn_topic_0000001872423361_p13118746101310"></a>valueHolder</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001872423361_p1611810464138"><a name="zh-cn_topic_0000001872423361_p1611810464138"></a><a name="zh-cn_topic_0000001872423361_p1611810464138"></a>保证KernelRunContextHolder内部值不析构的智能指针</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001872423361_row121189469136"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001872423361_p3118646181314"><a name="zh-cn_topic_0000001872423361_p3118646181314"></a><a name="zh-cn_topic_0000001872423361_p3118646181314"></a>computeNodeExtendHolder</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001872423361_p73531246172518"><a name="zh-cn_topic_0000001872423361_p73531246172518"></a><a name="zh-cn_topic_0000001872423361_p73531246172518"></a>可转化成ComputeNodeInfo类的智能指针</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001872423361_row1367219464226"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001872423361_p7673164602210"><a name="zh-cn_topic_0000001872423361_p7673164602210"></a><a name="zh-cn_topic_0000001872423361_p7673164602210"></a>context</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001872423361_p56731046172212"><a name="zh-cn_topic_0000001872423361_p56731046172212"></a><a name="zh-cn_topic_0000001872423361_p56731046172212"></a>指向KernelRunContext类的指针</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="zh-cn_topic_0000001872423361_zh-cn_topic_0000001441184464_section633mcpsimp"></a>

无

## 调用示例<a name="zh-cn_topic_0000001872423361_zh-cn_topic_0000001441184464_section642mcpsimp"></a>

```
auto holder = context_ascendc::ContextBuilder().Inputs().Outputs().BuildKernelRunContext();
if (holder != nullptr) {
    gert::KernelContext* tilingParseContext = holder->GetContext<gert::KernelContext>();
    gert::ComputeNodeInfo* info = holder->MutableComputeNodeInfo();
}
```

