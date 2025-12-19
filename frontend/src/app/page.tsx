"use client";

import { useEffect, useState, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Activity, Zap, Shield, Database, Clock, Cpu,
  ArrowUpRight, TrendingUp, Layers, ZapOff
} from "lucide-react";
import { Sidebar } from "@/components/Sidebar";
import { createClient } from "@/utils/supabase";

interface ReconstructionLog {
  total_lines: number;
  selected_lines: number;
  overhead_ms: number;
  sequence: Array<{
    source: string;
    text: string;
    score: number;
    line_index: number;
  }>;
}

interface RequestLog {
  id: string;
  session_id: string;
  model: string;
  tokens_in: number;
  tokens_out: number;
  tokens_saved: number;
  latency_ms: number;
  cost_saved_usd?: number;
  total_cost_usd?: number;
  reconstruction_log?: ReconstructionLog;
  timestamp: number;
}

// ... (Stats interface and component setup remain same) ...


interface Stats {
  total_tokens_saved: number;
  total_requests: number;
  recent_requests: RequestLog[];
}

export default function Dashboard() {
  const [stats, setStats] = useState<Stats | null>(null);
  const [logs, setLogs] = useState<RequestLog[]>([]);
  const [connected, setConnected] = useState(false);
  const [expandedLogId, setExpandedLogId] = useState<string | null>(null);
  const ws = useRef<WebSocket | null>(null);

  const [apiKey, setApiKey] = useState<string>("");
  const GATEWAY_URL = process.env.NEXT_PUBLIC_GATEWAY_URL || "http://localhost:8000";
  const WS_URL = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000";
  const supabase = createClient();

  useEffect(() => {
    const init = async () => {
      // 1. Try to get key from localStorage
      let key = typeof window !== 'undefined' ? localStorage.getItem("v1_key") : null;

      // 2. If no key, fetch from Supabase
      if (!key) {
        const { data: { user } } = await supabase.auth.getUser();
        if (user) {
          try {
            const resp = await fetch(`${GATEWAY_URL}/api/keys`, {
              headers: { "x-user-id": user.id }
            });
            if (resp.ok) {
              const keys = await resp.json();
              if (Array.isArray(keys) && keys.length > 0) {
                // Use the most recently created key
                key = keys[0].raw_key;
                localStorage.setItem("v1_key", key!);
              }
            }
          } catch (e) {
            console.error("Failed to fetch keys:", e);
          }
        }
      }

      setApiKey(key || "");
    };
    init();
  }, []);

  useEffect(() => {
    if (!apiKey) return;

    let socket: WebSocket | null = null;
    let reconnectTimeout: NodeJS.Timeout;

    const connect = () => {
      if (socket?.readyState === WebSocket.OPEN) return;

      console.log(`Connecting to WS with key: ${apiKey}`);
      socket = new WebSocket(`${WS_URL}/ws/${apiKey}`);

      socket.onopen = () => {
        console.log("WS Connected");
        setConnected(true);
      };

      socket.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          console.log("WS Message:", message.type);

          if (message.type === "init") {
            setStats(message.data);
            setLogs(message.data.recent_requests || []);
          } else if (message.type === "request") {
            const newLog = message.data;
            setLogs((prev) => {
              const exists = prev.find(l => l.id === newLog.id);
              if (exists) return prev;
              return [newLog, ...prev].slice(0, 50);
            });
            setStats((prev) => prev ? {
              ...prev,
              total_tokens_saved: prev.total_tokens_saved + newLog.tokens_saved,
              total_requests: prev.total_requests + 1
            } : null);
          }
        } catch (e) {
          console.error("WS Parse Error:", e);
        }
      };

      socket.onclose = (e) => {
        console.log("WS Closed:", e.code, e.reason);
        setConnected(false);
        // Only reconnect if not 4003 (invalid key)
        if (e.code !== 4003) {
          reconnectTimeout = setTimeout(connect, 3000);
        }
      };

      socket.onerror = (e) => {
        console.error("WS Error:", e);
      };

      ws.current = socket;
    };

    connect();

    return () => {
      clearTimeout(reconnectTimeout);
      if (socket) socket.close();
    };
  }, [apiKey]);

  const totalCostSaved = logs.reduce((acc, log) => acc + (log.cost_saved_usd || 0), 0);

  return (
    <div className="layout-wrapper">
      <Sidebar />

      <main className="main-content">
        {/* Top Header */}
        <header style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end", marginBottom: "48px" }}>
          <div>
            <h1 style={{ fontSize: "24px", marginBottom: "4px" }}>Overview</h1>
            <p className="text-muted" style={{ fontSize: "14px" }}>Real-time performance and cost optimization metrics.</p>
          </div>
          <div className="glass" style={{ padding: "8px 16px", display: "flex", alignItems: "center", gap: "12px" }}>
            <div className={connected ? "live-indicator" : ""} style={{ background: connected ? "var(--accent)" : "#ff4444" }} />
            <span style={{ fontSize: "11px", fontWeight: 600 }}>{connected ? "GATEWAY ONLINE" : "GATEWAY OFFLINE"}</span>
          </div>
        </header>

        {/* Stats Grid */}
        <div className="stats-grid">
          <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="glass card stat-item">
            <div className="stat-label"><Zap size={12} style={{ verticalAlign: "middle", marginRight: "6px" }} /> Est. Cost Saved</div>
            <div className="stat-value text-accent">${totalCostSaved.toFixed(4)}</div>
            <div style={{ fontSize: "11px", color: "rgba(0,255,136,0.5)", display: "flex", alignItems: "center", gap: "4px" }}>
              <TrendingUp size={12} /> Based on current usage
            </div>
          </motion.div>

          <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }} className="glass card stat-item">
            <div className="stat-label"><Activity size={12} style={{ verticalAlign: "middle", marginRight: "6px" }} /> Total Requests</div>
            <div className="stat-value">{(stats?.total_requests || 0).toLocaleString()}</div>
            <div style={{ fontSize: "11px", opacity: 0.4 }}>Across {new Set(logs.map(l => l.session_id)).size} active sessions</div>
          </motion.div>

          <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }} className="glass card stat-item">
            <div className="stat-label"><Shield size={12} style={{ verticalAlign: "middle", marginRight: "6px" }} /> Avg. Latency</div>
            <div className="stat-value">
              {logs.length > 0
                ? (logs.reduce((acc, l) => acc + l.latency_ms, 0) / logs.length).toFixed(0)
                : "0"}
              <span style={{ fontSize: "14px", opacity: 0.4, marginLeft: "4px" }}>ms</span>
            </div>
            <div style={{ fontSize: "11px", opacity: 0.4 }}>
              P95: {logs.length > 0 ? [...logs].sort((a, b) => b.latency_ms - a.latency_ms)[Math.floor(logs.length * 0.05)]?.latency_ms.toFixed(0) : "0"}ms
            </div>
          </motion.div>

          <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 }} className="glass card stat-item">
            <div className="stat-label"><Layers size={12} style={{ verticalAlign: "middle", marginRight: "6px" }} /> Optimization Rate</div>
            <div className="stat-value">
              {logs.length > 0
                ? ((logs.reduce((acc, l) => acc + l.tokens_saved, 0) / (logs.reduce((acc, l) => acc + l.tokens_in, 0) || 1)) * 100).toFixed(1)
                : "0.0"}%
            </div>
            <div style={{ fontSize: "11px", opacity: 0.4 }}>V1 Context Savings</div>
          </motion.div>
        </div>

        {/* Live Feed Section */}
        <section className="glass" style={{ padding: "0", overflow: "hidden" }}>
          <div className="card-header" style={{ padding: "20px 24px", borderBottom: "1px solid var(--border)" }}>
            <h2 style={{ fontSize: "14px", fontWeight: 600 }}>Live Request Stream</h2>
            <div className="btn btn-outline" style={{ fontSize: "10px", padding: "4px 10px" }}>
              Clear Logs
            </div>
          </div>

          <div className="log-list">
            <div className="log-item log-header" style={{ gridTemplateColumns: "2fr 1fr 1fr 1fr 1fr 1fr" }}>
              <div>Session ID</div>
              <div>Model</div>
              <div>Tokens</div>
              <div>Savings</div>
              <div>Cost Saved</div>
              <div style={{ textAlign: "right" }}>Time</div>
            </div>

            <AnimatePresence initial={false}>
              {logs.map((log) => (
                <motion.div
                  key={log.id}
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: "auto" }}
                  exit={{ opacity: 0 }}
                  style={{ borderBottom: "1px solid var(--border)" }}
                >
                  <div
                    className="log-item"
                    style={{
                      gridTemplateColumns: "2fr 1fr 1fr 1fr 1fr 1fr",
                      cursor: "pointer",
                      background: expandedLogId === log.id ? "rgba(255,255,255,0.02)" : "transparent"
                    }}
                    onClick={() => setExpandedLogId(expandedLogId === log.id ? null : log.id)}
                  >
                    <div style={{ display: "flex", alignItems: "center", gap: "10px", overflow: "hidden" }}>
                      <div style={{ width: "4px", height: "4px", background: "var(--accent)", borderRadius: "50%", flexShrink: 0 }} />
                      <span style={{ opacity: 0.9, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }} title={log.session_id}>
                        {log.session_id}
                      </span>
                    </div>
                    <div className="text-muted" style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                      <Cpu size={12} /> {log.model}
                    </div>
                    <div className="text-muted">
                      {log.tokens_in + log.tokens_out}
                    </div>
                    <div style={{ color: log.tokens_saved > 0 ? "var(--accent)" : "rgba(255,255,255,0.2)" }}>
                      {log.tokens_saved > 0 ? (
                        <span style={{ display: "flex", alignItems: "center", gap: "4px" }}>
                          <Zap size={12} /> {log.tokens_saved.toLocaleString()}
                        </span>
                      ) : "-"}
                    </div>
                    <div style={{ color: "var(--accent)" }}>
                      ${(log.cost_saved_usd || 0).toFixed(5)}
                    </div>
                    <div style={{ textAlign: "right", opacity: 0.4 }}>
                      {new Date(log.timestamp * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
                    </div>
                  </div>

                  {expandedLogId === log.id && (
                    <div style={{ padding: "24px", background: "rgba(0,0,0,0.3)", fontSize: "13px", borderTop: "1px solid var(--border)" }}>
                      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr 1fr", gap: "24px", marginBottom: "32px" }}>
                        <div>
                          <div className="text-muted" style={{ marginBottom: "4px" }}>Full Session ID</div>
                          <div style={{ fontFamily: "monospace", fontSize: "12px", color: "var(--fg)" }}>{log.session_id}</div>
                        </div>
                        <div>
                          <div className="text-muted" style={{ marginBottom: "4px" }}>Latency / Overhead</div>
                          <div style={{ color: "var(--fg)" }}>
                            {log.latency_ms.toFixed(0)}ms
                            {log.reconstruction_log?.overhead_ms ? <span className="text-muted" style={{ marginLeft: "6px" }}>({log.reconstruction_log.overhead_ms.toFixed(1)}ms opt)</span> : null}
                          </div>
                        </div>
                        <div>
                          <div className="text-muted" style={{ marginBottom: "4px" }}>Total Cost</div>
                          <div style={{ color: "var(--fg)" }}>${(log.total_cost_usd || 0).toFixed(6)}</div>
                        </div>
                        <div>
                          <div className="text-muted" style={{ marginBottom: "4px" }}>Context Compression</div>
                          <div style={{ color: "var(--fg)" }}>
                            {log.reconstruction_log?.total_lines
                              ? `${log.reconstruction_log.selected_lines} / ${log.reconstruction_log.total_lines} lines kept`
                              : "N/A"}
                          </div>
                        </div>
                      </div>

                      {log.reconstruction_log?.sequence && log.reconstruction_log.sequence.length > 0 && (
                        <div>
                          <h3 style={{ fontSize: "12px", fontWeight: 600, marginBottom: "16px", color: "rgba(255,255,255,0.4)" }}>CONTEXT RECONSTRUCTION SEQUENCE</h3>
                          <div className="sequence-container">
                            {log.reconstruction_log.sequence.map((item, idx) => (
                              <motion.div
                                key={idx}
                                initial={{ opacity: 0, x: -10 }}
                                animate={{ opacity: 1, x: 0 }}
                                transition={{ delay: idx * 0.05 }}
                                className={`sequence-card ${item.score > 0 ? "optimized" : ""}`}
                              >
                                <div style={{ width: "32px", fontSize: "10px", opacity: 0.3, fontFamily: "monospace" }}>
                                  {item.line_index.toString().padStart(3, '0')}
                                </div>
                                <div style={{ width: "60px" }}>
                                  <span className={`sequence-tag ${item.source === 'user' ? 'tag-user' : 'tag-assistant'}`}>
                                    {item.source?.slice(0, 3)}
                                  </span>
                                </div>
                                <div style={{ flex: 1, fontSize: "12px", opacity: 0.9, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                                  {item.text}
                                </div>
                                <div style={{ width: "80px", textAlign: "right" }}>
                                  <div style={{ fontSize: "10px", color: item.score > 0 ? "var(--accent)" : "rgba(255,255,255,0.2)" }}>
                                    {item.score > 0 ? (
                                      <span style={{ display: "flex", alignItems: "center", justifyContent: "flex-end", gap: "4px" }}>
                                        <TrendingUp size={10} /> {(item.score * 100).toFixed(0)}%
                                      </span>
                                    ) : (
                                      "Dropped"
                                    )}
                                  </div>
                                </div>
                              </motion.div>
                            ))}
                            {log.reconstruction_log.total_lines > log.reconstruction_log.sequence.length && (
                              <div style={{ textAlign: "center", padding: "12px", fontSize: "11px", opacity: 0.3 }}>
                                + {log.reconstruction_log.total_lines - log.reconstruction_log.sequence.length} lines distilled out
                              </div>
                            )}
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </motion.div>
              ))}
            </AnimatePresence>

            {logs.length === 0 && (
              <div style={{ padding: "60px", textAlign: "center", background: "var(--bg)" }}>
                <Activity size={32} className="text-muted" style={{ marginBottom: "16px", opacity: 0.2 }} />
                <p className="text-muted" style={{ fontSize: "13px" }}>Listening for incoming requests on <code style={{ color: "var(--accent)" }}>:8000</code>...</p>
              </div>
            )}
          </div>
        </section>
      </main>
    </div>
  );
}
