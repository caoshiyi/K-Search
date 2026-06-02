# ge::graphStatus

**页面ID:** atlasopapi_07_00513  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00513.html

---

graphStatus类型即uint32_t类型，其不同的状态说明如下：

| 状态 | 值 | 说明 |
| --- | --- | --- |
| ge::GRAPH_SUCCESS | 0 | 对应函数执行成功。 |
| ge::GRAPH_FAILED | 0xFFFFFFFF | 对应函数执行失败。 |
| ge::GRAPH_PARAM_INVALID | 50331649 | 对应函数执行失败，执行时存在参数无法通过校验的情况。 |
| ge::GRAPH_NOT_CHANGED | 1343242304 | 不符合常量折叠的条件时的错误码。 |
| ge::GRAPH_NODE_WITHOUT_CONST_INPUT | 50331648 | 检测到网络中的某个算子的输入没有const数据的场景。 |
| ge::GRAPH_NODE_NEED_REPASS | 50331647 | 图节点需要被重新处理或遍历，常见于infershape阶段。 |
| ge::GRAPH_INVALID_IR_DEF | 50331646 | 无效的IR定义。 |
| ge::OP_WITHOUT_IR_DATATYPE_INFER_RULE | 50331645 | 算子缺少IR数据类型推断规则。 |
| ge::GRAPH_PARAM_OUT_OF_RANGE | 50331644 | 函数参数的值超出了允许的有效范围。 |
| ge::GRAPH_MEM_OPERATE_FAILED | 50331539 | 内存操作（如分配或释放）失败。 |
| ge::GRAPH_NULL_PTR | 50331538 | 函数内部遇到了非法的空指针。 |
| ge::GRAPH_MEMCPY_FAILED | 50331537 | 内存拷贝操作失败。 |
| ge::GRAPH_MEMSET_FAILED | 50331536 | 内存设置操作失败。 |
| ge::GRAPH_MATH_CAL_FAILED | 50331429 | 数学计算（如卷积、矩阵乘）失败。 |
| ge::GRAPH_ADD_OVERFLOW | 50331428 | 加法运算结果溢出。 |
| ge::GRAPH_MUL_OVERFLOW | 50331427 | 乘法运算结果溢出。 |
| ge::GRAPH_RoundUp_Overflow | 50331426 | 向上取整运算导致溢出。 |
