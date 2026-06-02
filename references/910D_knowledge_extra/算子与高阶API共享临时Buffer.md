# 算子与高阶API共享临时Buffer<a name="ZH-CN_TOPIC_0000002523129140"></a>

【优先级】高

【描述】如果算子使用的高阶API需要传入临时Buffer，如SoftMax，该临时空间会挤占算子其他计算的空间，从而导致单次计算搬运的数据量变少，搬运的次数变多。此场景可通过共享临时Buffer空间，提升单次搬运的数据量，减少搬运的次数，提升内存使用效率。

【反例】

SoftMax高阶API计算需要临时Buffer空间，算子在进行其他计算时拥有独立临时Buffer。UB空间是固定的，假设可以给SoftMax和Add能分配临时空间为64KB，SoftMax计算需要的临时Buffer空间tmpSoftmaxBuf占用32KB，则存储Add计算结果的LocalTensor tmpSumBuf最多只能分配32KB。如果src0Tensor计算的数据量是512KB，则需要搬运512 / 32 =  **16**次。

```
...
constexpr int32_t blockLen = 32 * 1024;
TBuf<TPosition::VECCALC> tmpSoftmaxBuf; 
pipe.InitBuffer(tmpSoftmaxBuf, softmaxBufSize * sizeof(uint8_t));  // 单独分配Softmax的临时Buf 32KB
TBuf<TPosition::VECCALC> tmpSumBuf;
pipe.InitBuffer(tmpSumBuf, sumBufSize * sizeof(T)); // 单独分配Add的临时Buf，且softmaxBufSize * sizeof(uint8_t) + sumBufSize * sizeof(T) <= 64KB
...
for (int i = 0; i < 16; i++) {
    ...
    LocalTensor<uint8_t> tmpSoftmaxTensor = tmpSoftmaxBuf.Get<uint8_t>(softmaxBufSize);
    SoftMax<T, true, true>(dstTensor, expSumTensor, dstMaxTensor, srcTensor, tmpSoftmaxTensor, tiling);
    ...
    DataCopy(src0Tensor, src0Gm[i * blockLen / sizeof(T)], Params);
    ...
    LocalTensor<T> tmpSumTensor = tmpSumBuf.Get<T>(sumBufSize);
    Add<T>(tmpSumTensor, src0Tensor, src1Tensor, count);
    ...
}
...
```

【正例】

SoftMax高阶API计算需要临时Buffer空间，算子在进行其他计算时可以共享此临时Buffer，按照上述假设只需要搬运512 / 64 =  **8**次。

```
...
constexpr int32_t blockLen = 64 * 1024;
TBuf<TPosition::VECCALC> tmpSharedBuf;
pipe.InitBuffer(tmpSharedBuf, bufferSize); // 共享分配bufferSize = MAX(softmaxBufSize * sizeof(uint8_t), sumBufSize * sizeof(T)) <= 64KB
...
for (int i = 0; i < 8; i++) {
    ...
    LocalTensor<uint8_t> tmpSharedTensor = tmpSharedBuf.Get<uint8_t>(softmaxBufSize);
    SoftMax<T, true, true>(dstTensor, expSumTensor, dstMaxTensor, srcTensor, tmpSharedTensor, tiling);
    ...
    DataCopy(src0Tensor, src0Gm[i * blockLen / sizeof(T)], Params);
    ...
    LocalTensor<T> tmpSumTensor = tmpSharedBuf.Get<T>(sumBufSize);
    Add<T>(tmpSumTensor, src0Tensor, src1Tensor, count);
    ...
}
...
```

