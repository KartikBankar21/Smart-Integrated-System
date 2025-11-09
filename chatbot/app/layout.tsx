import Link from "next/link"
import React, { ReactNode } from "react";


export const metadata = {
  title: "Smart Integrated System",
  description: "AI-powered multi-module platform"
}

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body style={{ fontFamily: "sans-serif", backgroundColor: "#f5f5f5" }}>
        <nav style={{ padding: "1rem", background: "#1e293b", color: "#fff" }}>
          <Link href="/" style={{ marginRight: "1rem", color: "white" }}>Home</Link>
          <Link href="/chatbot" style={{ marginRight: "1rem", color: "white" }}>Chatbot</Link>
          <Link href="/billing" style={{ marginRight: "1rem", color: "white" }}>Billing</Link>
          <Link href="/delivery" style={{ color: "white" }}>Delivery</Link>
        </nav>
        <main style={{ padding: "2rem" }}>{children}</main>
      </body>
    </html>
  )
}
