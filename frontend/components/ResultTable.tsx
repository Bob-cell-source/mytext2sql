import type { CSSProperties } from "react";

type Props = {
  columns: string[];
  rows: Record<string, unknown>[];
};


export default function ResultTable({ columns, rows }: Props) {
  if (!columns.length) {
    return <div style={{ color: "#667085" }}>无结果行</div>;
  }

  return (
    <div style={{ overflowX: "auto", border: "1px solid #e4ddcf", borderRadius: 12 }}>
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 14 }}>
        <thead style={{ background: "#f6f2e8" }}>
          <tr>
            {columns.map((column) => (
              <th key={column} style={thStyle}>
                {column}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, idx) => (
            <tr key={idx}>
              {columns.map((column) => (
                <td key={column} style={tdStyle}>
                  {String(row[column] ?? "")}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}


const thStyle: CSSProperties = {
  textAlign: "left",
  padding: 12,
  borderBottom: "1px solid #e4ddcf"
};

const tdStyle: CSSProperties = {
  padding: 12,
  borderBottom: "1px solid #f0eadf",
  verticalAlign: "top"
};
