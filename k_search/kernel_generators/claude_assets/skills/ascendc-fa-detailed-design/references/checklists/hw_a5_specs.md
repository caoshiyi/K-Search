# A5 硬件规格参考

FA 详细设计中涉及的 A5（arch35/351x/950）硬件参数。

---

## 存储层次

| 存储层 | 容量 | 带宽 | 访问方式 | 备注 |
|--------|------|------|---------|------|
| HBM (GM) | 大容量 | ~2TB/s | DMA | 全局存储 |
| L1 | 512KB | 高 | DMA (GM↔L1) | 数据预加载层 |
| L0A | 64KB | - | L1→L0A | Cube 左操作数 |
| L0B | 64KB | - | L1→L0B | Cube 右操作数 |
| L0C | 256KB | - | Cube 输出 | Cube 结果暂存 |
| UB | 248KB | - | L0C→UB, Vec R/W | Vector 计算主存 |
| VecReg | 256×256B | - | UB→Reg→计算 | Regbase 寄存器（A5 新增） |

## 与 A3（910B）的对比

| 参数 | A3（910B/arch32/2201） | A5（950/arch35/5102） |
|------|----------------------|---------------------|
| UB | 192KB（184KB Vec 可用） | 248KB |
| L1 | 512KB - 128B | 512KB |
| L0C | 128KB | 256KB |
| L0A/L0B | 64KB | 64KB |
| CV 比例 | 1:1 或 1:2 | 1:2（固定） |
| UB Bank 结构 | 16 group × 3 bank × 4KB | 8 group × 2 bank × 16KB |
| Vector 编程模型 | Membase (SIMD) | Regbase (SIMD/SIMT) |
| Vector 寄存器 | 无显式寄存器 | 256 VecReg × 256B |
| 2×VL 模式 | 不支持 | 支持（寄存器数减半） |
| Subnormal 处理 | 硬件支持 | 软件仿真（需配置） |
| __NPU_ARCH__ | 2201 | 5102 (DAV_3510) |

## Regbase 编程模型要点

### 数据流
```
GM → L1 → UB → Register → 计算 → Register → UB → GM
                ↑ LoadAlign          StoreAlign ↓
```

### VF 函数规范
```cpp
// __simd_vf__ 修饰，所有 Scalar 计算在外部完成
__simd_vf__ void MyVF(T* dst, const T* src, uint32_t calCount,
                       uint16_t repeatTimes, uint16_t oneRepSize) {
    MicroAPI::MaskReg mask;
    MicroAPI::RegTensor<T> dstVreg;
    MicroAPI::RegTensor<T> srcVreg;

    for (uint16_t i = 0; i < repeatTimes; ++i) {
        mask = MicroAPI::UpdateMask<T>(calCount);
        MicroAPI::LoadAlign(srcVreg, src + i * oneRepSize);
        // ... 计算 ...
        MicroAPI::StoreAlign(dst + i * oneRepSize, dstVreg, mask);
    }
}

// 外部调用
constexpr uint16_t oneRepSize = GetVecLen() / sizeof(T);  // 256B / sizeof(T)
uint16_t repeatTimes = CeilDivision(calCount, oneRepSize);
SIMD_VF(MyVF, dst, src, calCount, repeatTimes, oneRepSize);
```

### 关键约束
1. **SIMD_VF 内禁止 Scalar 计算** — 会阻塞 Vector 流水
2. **寄存器申请顺序**：MaskReg → dst RegTensor → src RegTensor
3. **Subnormal 需配置**：涉及 exp/log/sqrt/rsqrt/div/reciprocal 时需设置计算模式
4. **Bank 冲突**：8 group × 2 bank × 16KB，同 bank group 的并发访问可能冲突
5. **AIV1 事件偏移**：CrossCore 事件 ID 对 AIV1 自动 +16

## 核间同步机制

### CrossCore 事件
```cpp
// AIC → AIV 通知
CrossCoreSetFlag(eventId);     // AIC 侧发送
CrossCoreWaitFlag(eventId);    // AIV 侧等待

// AIV → AIC 通知
CrossCoreSetFlag(eventId);     // AIV 侧发送
CrossCoreWaitFlag(eventId);    // AIC 侧等待
```

### 事件 ID 规则
- AIV0 和 AIC 共享一套事件 ID 空间
- AIV1 的事件 ID = AIV0 事件 ID + 16
- 同一事件不能被多个消费者同时 Wait

### Ping-Pong 同步
奇偶迭代使用不同事件集合，避免上一轮的 SetFlag 干扰本轮的 WaitFlag：
```
iter 0 (ping): 使用 eventId_set_A
iter 1 (pong): 使用 eventId_set_B
iter 2 (ping): 使用 eventId_set_A（已被 iter 0 消费完毕）
```

## FA 算子的 Cube 使用模式

### MatMul 配置
```cpp
// BMM1: Q × K^T → S
matmul.SetTensorA(Q_l0a);       // [Sq, D] 左操作数
matmul.SetTensorB(K_l0b, true); // [Sk, D] 转置，变为 [D, Sk]
matmul.Iterate(S_l0c);          // [Sq, Sk] 输出到 L0C

// BMM2: P × V → O
matmul.SetTensorA(P_l0a);       // [Sq, Sk] 左操作数
matmul.SetTensorB(V_l0b);       // [Sk, D] 右操作数
matmul.Iterate(O_l0c);          // [Sq, D] 输出到 L0C
```

### FixPipe（L0C → UB）
```cpp
// 将 Cube 输出从 L0C 搬到 UB，供 Vector 使用
fixpipe.SetSrc(S_l0c);
fixpipe.SetDst(S_ub);
fixpipe.Fixpipe();
```
