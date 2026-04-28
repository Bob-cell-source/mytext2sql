"use client";

import type { CSSProperties } from "react";
import { useEffect, useState } from "react";
import AskForm from "@/components/AskForm";
import SqlResult from "@/components/SqlResult";
import { askQuestion, fetchModels, type AskResponse, type ModelOption } from "@/lib/api";


export default function HomePage() {
  const [models, setModels] = useState<ModelOption[]>([]);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<AskResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [thinkingSeconds, setThinkingSeconds] = useState(0);

  useEffect(() => {
    fetchModels().then(setModels).catch((err) => setError(String(err)));
  }, []);

  useEffect(() => {
    if (!loading) {
      return;
    }
    const timer = window.setInterval(() => {
      setThinkingSeconds((prev) => prev + 1);
    }, 1000);
    return () => window.clearInterval(timer);
  }, [loading]);

  async function handleSubmit(payload: {
    question: string;
    modelId: string;
    mode: "native" | "intelligent";
    executeSql: boolean;
    knowledge: string;
    tableList: string[];
  }) {
    setLoading(true);
    setError(null);
    setThinkingSeconds(0);
    try {
      const response = await askQuestion({
        question: payload.question,
        model_id: payload.modelId,
        mode: payload.mode,
        execute_sql: payload.executeSql,
        knowledge: payload.knowledge,
        table_list: payload.tableList
      });
      setResult(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }

  return (
    <main style={styles.page}>
      <div style={styles.hero}>
        <div style={styles.heroText}>
          <div style={styles.eyebrow}>Text2SQL Platform</div>
          <h1 style={styles.headline}>原生方法与智能数据查询两种模式统一入口</h1>
          <p style={styles.subhead}>
            原生方法直接调用模型生成 SQL。智能数据查询会优先走你的 final_code.py 框架，注入 schema、knowledge、few-shot 和执行反馈链路。
          </p>
        </div>
      </div>

      <section style={styles.grid}>
        <AskForm models={models} loading={loading} thinkingSeconds={thinkingSeconds} onSubmit={handleSubmit} />
        <SqlResult result={result} error={error} />
      </section>
    </main>
  );
}


const styles: Record<string, CSSProperties> = {
  page: {
    maxWidth: 1280,
    margin: "0 auto",
    padding: "40px 20px 60px"
  },
  hero: {
    marginBottom: 24,
    padding: "32px 28px",
    borderRadius: 28,
    background: "radial-gradient(circle at top left, #f5d68a 0%, #f3e7c7 35%, #d8efe5 100%)"
  },
  heroText: {
    maxWidth: 760
  },
  eyebrow: {
    fontSize: 12,
    letterSpacing: 1.2,
    textTransform: "uppercase",
    color: "#755620",
    marginBottom: 8
  },
  headline: {
    margin: 0,
    fontSize: 42,
    lineHeight: 1.08
  },
  subhead: {
    fontSize: 16,
    color: "#4b5565",
    maxWidth: 640
  },
  grid: {
    display: "grid",
    gridTemplateColumns: "1.05fr 0.95fr",
    gap: 20
  }
};
