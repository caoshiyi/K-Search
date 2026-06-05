# 知识沉淀完成标记格式规范

## 文件位置

`current_task/artifacts/knowledge_curation_done.md`

## 格式模板

```markdown
# Knowledge Curation Done

- 时间：{YYYY-MM-DD HH:MM:SS}
- 报告：knowledge/reports/{task}_{date}_curation.md
- AscendC 调试回合数：{n}
- AscendC 技术发现数：{n}
- 新增候选数：{n}
- 更新条目数：{n}
```

## 填写说明

### 时间

知识沉淀完成的时间戳，格式为 `YYYY-MM-DD HH:MM:SS`。

### 报告

本次沉淀报告的路径，格式为 `knowledge/reports/{task}_{date}_curation.md`。

### AscendC 调试回合数

从 `debug_packet.json` 中统计的调试回合数。

### AscendC 技术发现数

从 `debug_log.md` 中统计的技术发现数。

### 新增候选数

本次任务中新增的候选 pattern 数量。

### 更新条目数

本次任务中更新的已有 pattern 数量。
