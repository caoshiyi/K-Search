# PostLiteral<a name="ZH-CN_TOPIC_0000002554424271"></a>

```
enum class PostLiteral {
    POST_MODE_NORMAL, // 正常场景，UB操作数地址不更新。LoadUnAlign针对连续非对齐搬入不支持POST_MODE_NORMAL模式。
    POST_MODE_UPDATE  // POST_MODE_UPDATE场景使用，UB地址同时作为输入和输出，每次调用会更新。
};
```

