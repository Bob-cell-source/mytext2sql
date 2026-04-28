import type { CSSProperties } from "react";
import type { AskResponse } from "@/lib/api";
import MarkdownBlock from "./MarkdownBlock";
import ResultTable from "./ResultTable";


type Props = {
  result: AskResponse | null;
  error: string | null;
};


export default function SqlResult({ result, error }: Props) {
  if (error) {
    return (
      <section style={styles.panel}>
        <h2>错误</h2>
        <MarkdownBlock title="Error" content={error} forceCode />
      </section>
    );
  }
  if (!result) {
    return (
      <section style={styles.panel}>
        <h2>结果</h2>
        <div style={{ color: "#667085" }}>提交问题后，这里会显示模型输出、SQL 和执行结果。</div>
      </section>
    );
  }

  return (
    <section style={styles.panel}>
      <div style={styles.metaRow}>
        <h2 style={{ margin: 0 }}>执行结果</h2>
        <div style={styles.badge}>{result.provider_mode}</div>
      </div>

      <MarkdownBlock title="Reasoning" content={result.reasoning || "模型未输出 reasoning"} />
      <MarkdownBlock title="Generated SQL / Solution" content={result.sql || "未提取到 SQL"} forceCode />
      <MarkdownBlock title="Raw Output" content={result.raw_output} />

      {result.execution && (
        <>
          <div style={styles.metaGrid}>
            <div><strong>Total:</strong> {formatMs(result.total_elapsed_ms)}</div>
            <div><strong>Model:</strong> {formatMs(result.model_elapsed_ms)}</div>
            <div><strong>Status:</strong> {result.execution.status}</div>
            <div><strong>Row Count:</strong> {result.execution.row_count}</div>
            <div><strong>SQL Exec:</strong> {formatMs(result.execution.elapsed_ms)}</div>
          </div>
          {result.execution.error_message ? (
            <MarkdownBlock title="Execution Error" content={result.execution.error_message} forceCode />
          ) : (
            <ResultTable columns={result.execution.columns} rows={result.execution.rows} />
          )}
        </>
      )}
    </section>
  );
}


const styles: Record<string, CSSProperties> = {
  panel: {
    background: "#fff",
    borderRadius: 20,
    border: "1px solid #e7dfcf",
    padding: 24,
    boxShadow: "0 16px 40px rgba(73, 53, 18, 0.08)"
  },
  metaRow: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 18
  },
  badge: {
    background: "#dff2ee",
    color: "#135c4f",
    padding: "6px 10px",
    borderRadius: 999,
    fontSize: 12
  },
  metaGrid: {
    display: "flex",
    gap: 20,
    marginBottom: 16,
    flexWrap: "wrap"
  }
};


function formatMs(ms: number) {
  if (!ms) {
    return "0 ms";
  }
  if (ms < 1000) {
    return `${ms} ms`;
  }
  return `${(ms / 1000).toFixed(2)} s`;
}
