import type { ReactNode } from "react";
import "./globals.css";
import type { Metadata } from "next";


export const metadata: Metadata = {
  title: "Text2SQL Playground",
  description: "Minimal frontend for Text2SQL model execution"
};


export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  );
}
