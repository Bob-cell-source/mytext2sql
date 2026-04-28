"use client";

import type { CSSProperties, ReactNode } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";


type Props = {
  title: string;
  content: string;
  forceCode?: boolean;
};


export default function MarkdownBlock({ title, content, forceCode = false }: Props) {
  return (
    <div style={{ marginBottom: 18 }}>
      <div style={styles.blockTitle}>{title}</div>
      <div style={styles.wrapper}>
        {forceCode ? (
          <pre style={styles.pre}>
            <code style={styles.code}>{content || ""}</code>
          </pre>
        ) : (
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            components={{
              p: ({ children }) => <p style={styles.p}>{children}</p>,
              ul: ({ children }) => <ul style={styles.ul}>{children}</ul>,
              ol: ({ children }) => <ol style={styles.ol}>{children}</ol>,
              li: ({ children }) => <li style={styles.li}>{children}</li>,
              strong: ({ children }) => <strong style={styles.strong}>{children}</strong>,
              em: ({ children }) => <em style={styles.em}>{children}</em>,
              pre: ({ children }) => <pre style={styles.pre}>{children}</pre>,
              code(props) {
                const { className, children } = props as {
                  className?: string;
                  children?: ReactNode;
                };
                return (
                  <code className={className} style={className ? styles.code : styles.inlineCode}>
                    {children}
                  </code>
                );
              }
            }}
          >
            {content || "_空内容_"}
          </ReactMarkdown>
        )}
      </div>
    </div>
  );
}


const styles: Record<string, CSSProperties> = {
  blockTitle: {
    fontWeight: 700,
    marginBottom: 8
  },
  wrapper: {
    background: "#fbf8f1",
    border: "1px solid #eee5d5",
    borderRadius: 12,
    padding: 14,
    overflowX: "auto"
  },
  p: {
    margin: "0 0 10px",
    lineHeight: 1.65,
    color: "#243142"
  },
  ul: {
    margin: "0 0 10px 18px",
    padding: 0
  },
  ol: {
    margin: "0 0 10px 18px",
    padding: 0
  },
  li: {
    marginBottom: 6,
    lineHeight: 1.6
  },
  strong: {
    color: "#102a43"
  },
  em: {
    color: "#52606d"
  },
  inlineCode: {
    background: "#efe8d8",
    color: "#7c2d12",
    padding: "2px 6px",
    borderRadius: 6,
    fontFamily: "SFMono-Regular, Consolas, monospace",
    fontSize: 13
  },
  pre: {
    margin: 0,
    padding: 16,
    borderRadius: 12,
    background: "#17212b",
    overflowX: "auto"
  },
  code: {
    color: "#d6e9ff",
    fontFamily: "SFMono-Regular, Consolas, monospace",
    fontSize: 13.5,
    lineHeight: 1.7,
    whiteSpace: "pre-wrap",
    wordBreak: "break-word"
  }
};
