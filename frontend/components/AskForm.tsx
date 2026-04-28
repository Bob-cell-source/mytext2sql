"use client";

import type { CSSProperties, FormEvent } from "react";
import { useState } from "react";
import type { ModelOption } from "@/lib/api";


type Props = {
  models: ModelOption[];
  loading: boolean;
  thinkingSeconds: number;
  onSubmit: (payload: {
    question: string;
    modelId: string;
    mode: "native" | "intelligent";
    executeSql: boolean;
    knowledge: string;
    tableList: string[];
  }) => Promise<void>;
};


export default function AskForm({ models, loading, thinkingSeconds, onSubmit }: Props) {
  const [question, setQuestion] = useState("");
  const [modelId, setModelId] = useState(models[0]?.model_id ?? "");
  const [mode, setMode] = useState<"native" | "intelligent">("intelligent");
  const [executeSql, setExecuteSql] = useState(true);
  const [knowledge, setKnowledge] = useState("");
  const [tableListText, setTableListText] = useState("");

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    await onSubmit({
      question,
      modelId,
      mode,
      executeSql,
      knowledge,
      tableList: tableListText
        .split(/[\n,，]+/)
        .map((item) => item.trim())
        .filter(Boolean)
    });
  }

  return (
    <form onSubmit={handleSubmit} style={styles.form}>
      <div style={styles.headerRow}>
        <div>
          <div style={styles.kicker}>Minimal Playground</div>
          <h1 style={styles.title}>数据查询执行台</h1>
        </div>
        <div style={styles.pill}>{models.length} models</div>
      </div>

      {loading && <div style={styles.timerBox}>已思考 <strong>{thinkingSeconds}s</strong></div>}

      <div style={styles.modeSwitch}>
        <button
          type="button"
          onClick={() => setMode("native")}
          style={mode === "native" ? { ...styles.modeButton, ...styles.modeButtonActive } : styles.modeButton}
        >
          原生方法
        </button>
        <button
          type="button"
          onClick={() => setMode("intelligent")}
          style={mode === "intelligent" ? { ...styles.modeButton, ...styles.modeButtonActive } : styles.modeButton}
        >
          智能数据查询
        </button>
      </div>

      <label style={styles.label}>
        问题
        <textarea
          style={styles.textarea}
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="输入自然语言问题"
          required
        />
      </label>

      <label style={styles.label}>
        业务知识
        <textarea
          style={styles.secondaryTextarea}
          value={knowledge}
          onChange={(e) => setKnowledge(e.target.value)}
          placeholder="可选。补充业务口径、特殊定义、时间范围、过滤逻辑"
        />
      </label>

      <label style={styles.label}>
        相关表
        <textarea
          style={styles.secondaryTextarea}
          value={tableListText}
          onChange={(e) => setTableListText(e.target.value)}
          placeholder="可选。每行一个表名，或用逗号分隔"
        />
      </label>

      <div style={styles.row}>
        <label style={styles.flexLabel}>
          模型
          <select style={styles.select} value={modelId} onChange={(e) => setModelId(e.target.value)}>
            {models.map((model) => (
              <option key={model.model_id} value={model.model_id}>
                {model.label}
              </option>
            ))}
          </select>
        </label>

        <label style={styles.checkboxLabel}>
          <input type="checkbox" checked={executeSql} onChange={(e) => setExecuteSql(e.target.checked)} />
          执行 SQL
        </label>
      </div>

      <button type="submit" disabled={loading || !modelId} style={styles.button}>
        {loading ? "运行中..." : "提交"}
      </button>
    </form>
  );
}


const styles: Record<string, CSSProperties> = {
  form: {
    background: "#fffdf8",
    border: "1px solid #e7dfcf",
    borderRadius: 20,
    padding: 24,
    boxShadow: "0 16px 40px rgba(73, 53, 18, 0.08)"
  },
  headerRow: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "flex-start",
    marginBottom: 20
  },
  kicker: {
    fontSize: 12,
    textTransform: "uppercase",
    letterSpacing: 1.2,
    color: "#8a6b32"
  },
  title: {
    margin: "6px 0 0",
    fontSize: 32
  },
  pill: {
    background: "#efe4c9",
    color: "#62451a",
    padding: "6px 12px",
    borderRadius: 999
  },
  label: {
    display: "block",
    fontWeight: 600,
    marginBottom: 16
  },
  textarea: {
    width: "100%",
    minHeight: 160,
    marginTop: 8,
    borderRadius: 14,
    border: "1px solid #d8ccb5",
    padding: 14,
    background: "#fff"
  },
  row: {
    display: "flex",
    gap: 16,
    alignItems: "center",
    marginBottom: 16,
    flexWrap: "wrap"
  },
  flexLabel: {
    display: "flex",
    flexDirection: "column",
    gap: 8,
    minWidth: 260,
    fontWeight: 600
  },
  select: {
    borderRadius: 12,
    border: "1px solid #d8ccb5",
    padding: "10px 12px",
    background: "#fff"
  },
  checkboxLabel: {
    display: "flex",
    gap: 8,
    alignItems: "center",
    marginTop: 24
  },
  button: {
    border: 0,
    borderRadius: 14,
    padding: "12px 18px",
    background: "#1f6f5f",
    color: "#fff",
    cursor: "pointer",
    fontWeight: 700
  },
  timerBox: {
    marginBottom: 16,
    background: "#eef7f5",
    border: "1px solid #cfe6de",
    color: "#135c4f",
    padding: "10px 12px",
    borderRadius: 12
  },
  modeSwitch: {
    display: "flex",
    gap: 10,
    marginBottom: 16,
    flexWrap: "wrap"
  },
  modeButton: {
    borderWidth: 1,
    borderStyle: "solid",
    borderColor: "#d8ccb5",
    background: "#fff",
    color: "#5f4a24",
    borderRadius: 999,
    padding: "10px 14px",
    cursor: "pointer",
    fontWeight: 600
  },
  modeButtonActive: {
    background: "#1f6f5f",
    borderWidth: 1,
    borderStyle: "solid",
    borderColor: "#1f6f5f",
    color: "#fff"
  },
  secondaryTextarea: {
    width: "100%",
    minHeight: 92,
    marginTop: 8,
    borderRadius: 14,
    border: "1px solid #d8ccb5",
    padding: 14,
    background: "#fff"
  }
};
